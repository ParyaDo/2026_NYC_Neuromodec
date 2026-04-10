"""Cross-modal CL + RL + safety framework for individualized TMS.

This module intentionally avoids importing heavy runtime dependencies at import time
so lightweight utilities/tests can run without full ML stack installed.
"""

from .config import FrameworkConfig
from .control import scale_action

__all__ = ["FrameworkConfig", "TMSFramework", "scale_action"]


def __getattr__(name: str):
    if name == "TMSFramework":
        from .pipeline import TMSFramework

        return TMSFramework
    raise AttributeError(name)
