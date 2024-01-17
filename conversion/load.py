import torch
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from typing import Optional, Tuple, List, Iterator 
import os, filelock, json, glob
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig
import marlin

# Adapted from https://github.com/vllm-project/vllm/blob/14cc317ba48229d93ee2417822d96ccb8db56abe/vllm/model_executor/weight_utils.py#L191

def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock

def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npcache":
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors

def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: Optional[bool] = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        load_format=load_format,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)

    if load_format == "npcache":
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, "w") as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()

@torch.no_grad()
def load_model(model_path):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path)

        if not hasattr(config, "quantization_config"):
            raise ValueError("Must be a Marlin quantized model, but your config has no quantization config.")
        if "quant_method" not in config.quantization_config:
            raise ValueError("Must be a Marlin quantized model, but your quantization config has no quant_method.")
        if config.quantization_config["quant_method"] != "marlin":
            raise ValueError(f"Must be a Marline model, but you passed a model with quant_method = {quant_method}")

        model = AutoModelForCausalLM.from_config(config)
        marlin.replace_linear(
            model.model, 
            groupsize=config.quantization_config["group_size"]
        )

    module_dict = dict(model.named_modules())
    for name, loaded_weight in hf_model_weights_iterator(model_path):
        module_name = ".".join(name.split(".")[:-1])
        param_name = name[len(module_name) + 1:]
        module = module_dict[module_name]

        if not hasattr(module, param_name):
            raise ValueError("Key mismatch.")
        
        setattr(module, param_name, torch.nn.Parameter(loaded_weight, requires_grad=False))

    return model