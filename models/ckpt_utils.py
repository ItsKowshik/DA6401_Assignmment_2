# models/ckpt_utils.py
import re

# Localizer head: reg_head.N -> head.head.M
_LOC_HEAD_MAP = {"2": "0", "5": "3", "8": "6"}

def remap_state_dict(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():

        # --- 1. Encoder: encoder.blockX.A.B.attr -> encoder.blockX.(A*3+B).attr ---
        m = re.match(r'^(encoder\.block\d+)\.(\d+)\.(\d+)\.(.+)$', k)
        if m:
            prefix, A, B, attr = m.groups()
            flat = int(A) * 3 + int(B)
            new_sd[f'{prefix}.{flat}.{attr}'] = v
            continue

        # --- 2. Localizer head: reg_head.N.attr -> head.head.M.attr ---
        m = re.match(r'^reg_head\.(\d+)\.(.+)$', k)
        if m:
            n, attr = m.groups()
            if n in _LOC_HEAD_MAP:
                new_sd[f'head.head.{_LOC_HEAD_MAP[n]}.{attr}'] = v
                continue
            # unexpected index — keep as-is, strict=False will drop it
            new_sd[k] = v
            continue

        # --- 3. Everything else (classifier head, unet decoder) — unchanged ---
        new_sd[k] = v

    return new_sd