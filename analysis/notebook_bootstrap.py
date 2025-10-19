"""Helper utilities for running project notebooks.

Importing this module ensures the repository root is available on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the repository root as the parent of this file's directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_project_root_on_path(root: Path = PROJECT_ROOT) -> Path:
    """Add the project root to sys.path if missing and return it."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


# Run on import so notebooks only need to `import analysis.notebook_bootstrap`.
ensure_project_root_on_path()

