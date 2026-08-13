"""Stellt sicher, dass Settings() auch ohne echten API-Key importierbar ist."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")
