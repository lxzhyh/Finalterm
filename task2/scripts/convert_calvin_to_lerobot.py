#!/usr/bin/env python3
"""
Entry point for CALVIN-to-LeRobot conversion.

Delegates to src.calvin_lerobot.convert module.

Usage:
    # Check LeRobot API first:
    python scripts/convert_calvin_to_lerobot.py --check_api

    # Convert with debug data:
    python scripts/convert_calvin_to_lerobot.py \
        --calvin_root third_party/calvin/dataset \
        --output_root data/lerobot_calvin \
        --splits debug \
        --max_episodes_per_env 5

    # Full conversion:
    python scripts/convert_calvin_to_lerobot.py \
        --calvin_root third_party/calvin/dataset \
        --output_root data/lerobot_calvin \
        --splits A ABC D
"""

import sys
from pathlib import Path

# Add src to path so calvin_lerobot package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calvin_lerobot.convert import main

if __name__ == "__main__":
    main()
