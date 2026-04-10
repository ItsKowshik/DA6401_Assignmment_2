import re

# Localizer: reg_head index → head.head index
_LOC_HEAD = {"2": "0", "5": "3", "8": "6"}

def _remap_vgg_state(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():

        # --- ENCODER: encoder.blockX.A.B.attr → encoder.blockX.(A*3+B).attr ---
        m = re.match(r'^(encoder\.block\d+)\.(\d+)\.(\d+)\.(.+)$', k)
        if m:
            prefix, A, B, attr = m.groups()
            new_sd[f'{prefix}.{int(A)*3 + int(B)}.{attr}'] = v
            continue

        # --- LOCALIZER HEAD: reg_head.N.attr → head.head.M.attr ---
        m = re.match(r'^reg_head\.(\d+)\.(.+)$', k)
        if m:
            n, attr = m.groups()
            if n in _LOC_HEAD:
                new_sd[f'head.head.{_LOC_HEAD[n]}.{attr}'] = v
            # else: unmapped index, drop it (strict=False handles it)
            continue

        # --- EVERYTHING ELSE: classifier head, unet decoder — unchanged ---
        new_sd[k] = v

    return new_sd