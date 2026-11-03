# -*- mode: python ; coding: utf-8 -*-

import os

a = Analysis(
    ['modellingScriptAssymetric.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_runtime_hook_qt.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# PyInstaller can collect incompatible ICU DLLs from its host PATH.  This Qt
# build correctly resolves ICU from Windows, so omit those conflicting copies.
a.binaries = [
    item for item in a.binaries
    if os.path.basename(item[0]).lower() not in {"icuuc.dll", "icudt78.dll"}
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ModellingScriptAssymetric',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
