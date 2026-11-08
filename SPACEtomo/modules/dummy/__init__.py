"""
Dummy module for SPACEtomo testing without SerialEM or a real microscope.

When DUMMY mode is active, call `install_mock_serialem()` to inject the mock
`serialem` module into `sys.modules` so that all `import serialem as sem`
statements throughout the codebase resolve to the mock implementation.

This should be called early, before any module that does `import serialem`.
"""

import sys


def install_mock_serialem():
    """Install the mock serialem module into sys.modules.

    After calling this, any `import serialem as sem` will get the mock.
    Safe to call multiple times (idempotent). Raises RuntimeError if the real serialem module is already loaded:
    DUMMY mode must never drive a microscope (modules imported earlier would keep using the real module).
    """
    from SPACEtomo.modules.dummy import serialem as _mock_sem
    current = sys.modules.get("serialem")
    if current is None:
        sys.modules["serialem"] = _mock_sem
    elif current is not _mock_sem:
        raise RuntimeError("The real SerialEM module is loaded, but DUMMY mode runs SPACEtomo on a simulated microscope only. "
                           "Set DUMMY = False (SPACEconfig dummy) to run on the microscope, or use 'SPACEtomo dummy' outside SerialEM.")


def uninstall_mock_serialem():
    """Remove the mock serialem module from sys.modules.

    Useful for test teardown when you want to restore the original state.
    """
    sys.modules.pop("serialem", None)


def reset_mock_state():
    """Reset all internal mock state. Useful between test cases."""
    if "serialem" in sys.modules:
        mod = sys.modules["serialem"]
        if hasattr(mod, "reset_mock_state"):
            mod.reset_mock_state()
