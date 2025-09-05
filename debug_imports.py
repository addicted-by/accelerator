#!/usr/bin/env python3
"""
Debug imports to find the issue.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

print("Starting import debug...")

try:
    print("1. Importing torch...")
    import torch

    print("✓ torch imported")

    print("2. Importing torch.nn...")
    import torch.nn as nn

    print("✓ torch.nn imported")

    print("3. Importing types...")
    from accelerator.hooks.types import HookType

    print("✓ HookType imported")

    print("4. Importing HookInfo...")
    from accelerator.hooks.types import HookInfo

    print("✓ HookInfo imported")

    print("5. Importing registry...")
    from accelerator.hooks.registry import HookRegistry

    print("✓ HookRegistry imported")

    print("6. Creating registry...")
    registry = HookRegistry()
    print("✓ Registry created")

    print("🎉 All imports successful!")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback

    traceback.print_exc()
