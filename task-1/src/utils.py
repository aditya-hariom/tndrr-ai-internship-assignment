"""
src/utils.py

Reproducibility and serialization utilities for LoRA fine-tuning and evaluation.
"""

import os
import json
import random
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def load_yaml_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads list of JSON objects from a .jsonl file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Saves records to a .jsonl file, creating parent directories if necessary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_json_payload(raw_text: str) -> Tuple[Optional[str], bool]:
    """
    Extracts JSON payload from model generation output.
    Returns:
        (extracted_json_str, was_raw_valid)
        where was_raw_valid indicates if the raw output was cleanly parsable without cleanup.
    """
    clean_text = raw_text.strip()
    # Check if raw text is directly parsable
    try:
        json.loads(clean_text)
        return clean_text, True
    except Exception:
        pass

    # Try extracting from markdown code block: ```json ... ``` or ``` ... ```
    md_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = md_pattern.search(clean_text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate, False
        except Exception:
            pass

    # Fallback: find outer matching braces { ... }
    first_brace = clean_text.find("{")
    last_brace = clean_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = clean_text[first_brace:last_brace + 1].strip()
        try:
            json.loads(candidate)
            return candidate, False
        except Exception:
            pass

    return None, False
