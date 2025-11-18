import os
import cv2
import json
import argparse
import numpy as np
import torch
import time
from doclayout_yolo import YOLOv10

def normalize(s: str) -> str:
    """Normalize class names for consistent matching"""
    return s.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")

ALIASES = {
    "text": "plain_text",
    "plain text": "plain_text",
    "plain-text": "plain_text",
    "figure-caption": "figure_caption",
    "table-caption": "table_caption",
    "table-footnote": "table_footnote",
    "formula-caption": "formula_caption",
    "isolated_formula": "isolate_formula",
    "isolated-formula": "isolate_formula",
    "isolated formula": "isolate_formula",
}

def map_alias(token: str) -> str:
    """Map class name aliases to canonical names"""
    t = token.strip().lower()
    return ALIASES.get(t, token)

def parse_keep_classes(names_dict, keep_arg: str):
    """Parse class names to keep, handling aliases and normalization"""
    # Create normalized mapping
    norm_model = {k: normalize(v) for k, v in names_dict.items()}
    inv = {}
    for k, v in norm_model.items():
        inv.setdefault(v, []).append(k)
    
    # Parse input argument
    tokens = [t.strip() for t in keep_arg.split(",") if t.strip()]
    ids, missing = [], []
    
    for raw in tokens:
        aliased = map_alias(raw)
        key = normalize(aliased)
        if key in inv:
            ids.extend(inv[key])
        else:
            missing.append(raw)
    
    if missing:
        available = sorted(set(names_dict.values()))
        print(f"⚠️ Warning: Requested classes not found: {missing}")
        print(f"   Available classes: {available}")
        # Return empty list instead of exiting - let caller decide
        return []
    
    return sorted(set(ids))

def get_class_id(names_dict, class_name):
    """Get class ID for a given class name, handling normalization"""
    norm_target = normalize(class_name)
    for cls_id, cls_name in names_dict.items():
        if normalize(cls_name) == norm_target:
            return cls_id
    return None

def load_image_fast(path):
    """Fast image loading with error handling"""
    try:
        # Use faster flags
        return cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"⚠️ Fast load failed for {path}: {e}")
        return None

def save_image_fast(path, img):
    """Fast image saving with optimized JPEG quality"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(path, img)
        return True
    except Exception as e:
        print(f"⚠️ Fast save failed for {path}: {e}")
        return False

# Note: The main() function is not needed anymore as rotate_detect.py handles everything directly
# But we keep these helper functions for compatibility

if __name__ == "__main__":
    print("ℹ️ This script is now used as a helper module. Run rotate_detect.py instead.")
    print("🚀 For best performance, use the optimized rotate_detect.py script.")
