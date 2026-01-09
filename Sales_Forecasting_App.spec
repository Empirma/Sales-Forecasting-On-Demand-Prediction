# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('/Users/yousefhatem/AOU - GP/Graduation Project FInal/data', 'data'), ('/Users/yousefhatem/AOU - GP/Graduation Project FInal/utils', 'utils')]
binaries = []
hiddenimports = ['sklearn.utils._weight_vector', 'joblib', 'pandas', 'numpy', 'matplotlib', 'statsmodels', 'holidays.countries', 'holidays.financial']
tmp_ret = collect_all('holidays')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app_qt.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sales_Forecasting_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Sales_Forecasting_App',
)
app = BUNDLE(
    coll,
    name='Sales_Forecasting_App.app',
    icon=None,
    bundle_identifier=None,
)
