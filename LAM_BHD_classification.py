from __future__ import annotations

import sys
from pathlib import Path

APP_CUSTOM_DIR = Path(__file__).resolve().parent / "app" / "custom"
sys.path.insert(0, str(APP_CUSTOM_DIR))

from train_single_site import main


if __name__ == "__main__":
    main()
