"""Prefer the Qt libraries packaged with the frozen application on Windows."""

import os
import sys
import ctypes


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    qt_bin_dir = os.path.join(sys._MEIPASS, "PyQt6", "Qt6", "bin")
    if os.path.isdir(qt_bin_dir):
        # QtCore.pyd must resolve against the matching Qt6 DLLs in this bundle,
        # rather than a potentially incompatible Qt installation on PATH.
        _qt_dll_directory_handle = os.add_dll_directory(qt_bin_dir)
        os.environ["PATH"] = qt_bin_dir + os.pathsep + os.environ.get("PATH", "")
        # Load the exact Qt core library before the PyQt extension module.  This
        # avoids binding QtCore.pyd to an incompatible Qt6Core.dll on PATH.
        _qt_core_library = ctypes.WinDLL(os.path.join(qt_bin_dir, "Qt6Core.dll"))
