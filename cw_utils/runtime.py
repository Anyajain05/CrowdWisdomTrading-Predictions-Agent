from __future__ import annotations

import site
import sys
from pathlib import Path


def ensure_user_site() -> None:
    project_root = Path(__file__).resolve().parent.parent
    vendor_dir = str(project_root / ".vendor")
    if Path(vendor_dir).exists() and vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
