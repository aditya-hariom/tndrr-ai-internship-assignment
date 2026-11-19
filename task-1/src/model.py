"""
src/model.py

Model loading and PEFT LoRA adapter injection for Qwen2.5-0.5B-Instruct.
Includes target module architecture verification and parameter accounting.
"""

from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

def get_device() -> torch.device:
    """Returns cuda device if available, otherwise cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads base model and tokenizer with appropriate precision and padding configuration."""
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Select dtype based on hardware
    if torch_dtype is None or torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base model: {model_name} with dtype={dtype}...")
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device_map:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map and not torch.cuda.is_available():
        model.to("cpu")

    return model, tokenizer

def verify_target_modules(model: PreTrainedModel, target_modules: list) -> bool:
    """
    Inspects model architecture to ensure specified target modules actually exist.
    Prevents silent failures where LoRA attaches to zero layers.
    """
    named_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in target_modules if m not in named_modules]
    if missing:
        raise ValueError(
            f"Target modules {missing} not found in model architecture. Available module names: {sorted(named_modules)}"
        )
    print(f"[Architecture Verified] All target modules {target_modules} found in model layers.")
    return True

def apply_lora(model: PreTrainedModel, lora_cfg: Dict[str, Any]) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Applies PEFT LoRA adapter to base model and reports parameter statistics.
    """
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    verify_target_modules(model, target_modules)

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM")
    )

    peft_model = get_peft_model(model, peft_config)

    trainable_params, all_param = peft_model.get_nb_trainable_parameters()
    pct_trainable = 100 * trainable_params / all_param

    param_stats = {
        "trainable_parameters": trainable_params,
        "total_parameters": all_param,
        "trainable_percentage": round(pct_trainable, 4)
    }

    print("=" * 60)
    print(f"LoRA Adapter Injected:")
    print(f"  * Trainable Parameters: {trainable_params:,}")
    print(f"  * Total Parameters:     {all_param:,}")
    print(f"  * Trainable %:          {pct_trainable:.4f}%")
    print("=" * 60)

    return peft_model, param_stats

def load_model_for_inference(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads base model (and optional LoRA adapter weights) for deterministic evaluation."""
    model, tokenizer = load_base_model_and_tokenizer(base_model_name)

    if adapter_path:
        print(f"Attaching LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    if device is None:
        target_device = get_device()
        model.to(target_device)

    return model, tokenizer
