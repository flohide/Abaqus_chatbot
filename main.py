#!/usr/bin/env python
"""Entry Point: Startet den Terminal-Chat.

Nutzung:
    python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_manual_bot.cli.chat import run_chat  # noqa: E402

if __name__ == "__main__":
    run_chat()
