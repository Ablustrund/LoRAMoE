import warnings
from copy import deepcopy

from packaging import version

from ..utils import logging
from .import_utils import importlib_metadata, is_accelerate_available, is_bitsandbytes_available


if is_bitsandbytes_available():
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters

logger = logging.get_logger(__name__)


def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, fp16_statistics=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for serialization.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    is_4bit = False
    is_8bit = False
    if is_buffer or not is_bitsandbytes_available():
        is_8bit = False
        is_4bit = False
    else:
        is_4bit = hasattr(bnb.nn, "Params4bit") and isinstance(module._parameters[tensor_name], bnb.nn.Params4bit)
        is_8bit = isinstance(module._parameters[tensor_name], bnb.nn.Int8Params)

    if is_8bit or is_4bit:
        param = module._parameters[tensor_name]
        if param.device.type != "cuda":
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to("cpu")
                if value.dtype == torch.int8:
                    is_8bit_serializable = version.parse(importlib_metadata.version("bitsandbytes")) > version.parse(
                        "0.37.2"
                    )
                    if not is_8bit_serializable:
                        raise ValueError(
                            "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                            "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                        )
            else:
                new_value = torch.tensor(value, device="cpu")

            kwargs = old_value.__dict__
            if is_8bit:
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(device)
            elif is_4bit:
                new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(device)

            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                setattr(module.weight, "SCB", fp16_statistics.to(device))

    else:
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value


def _replace_with_bnb_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, has_been_replaced=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    if quantization_config.quantization_method() == "llm_int8":
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                        has_been_replaced = True
                    else:
                        if (
                            quantization_config.llm_int8_skip_modules is not None
                            and name in quantization_config.llm_int8_skip_modules
                        ):
                            pass
                        else:
                            model._modules[name] = bnb.nn.Linear4bit(
                                module.in_features,
                                module.out_features,
                                module.bias is not None,
                                quantization_config.bnb_4bit_compute_dtype,
                                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                                quant_type=quantization_config.bnb_4bit_quant_type,
                            )
                            has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


# For backward compatibility
def replace_8bit_linear(*args, **kwargs):
    warnings.warn(
        "`replace_8bit_linear` will be deprecated in a future version, please use `replace_with_bnb_linear` instead",
        FutureWarning,
    )
    return replace_with_bnb_linear(*args, **kwargs)


# For backward compatiblity
def set_module_8bit_tensor_to_device(*args, **kwargs):
    warnings.warn(
        "`set_module_8bit_tensor_to_device` will be deprecated in a future version, please use `set_module_quantized_tensor_to_device` instead",
        FutureWarning,
    )
    return set_module_quantized_tensor_to_device(*args, **kwargs)


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = list(tied_params.values())
    else:
        tied_keys = sum([x[1:] for x in tied_params], [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = tied_keys + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names
