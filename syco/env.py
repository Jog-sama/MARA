from __future__ import annotations

import os


def load_env(path: str = ".env") -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path, os.path.join(root, ".env")]
    loaded = {}
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                loaded[key] = val
                # keeping values already exported in the shell
                os.environ.setdefault(key, val)
        break
    return loaded
