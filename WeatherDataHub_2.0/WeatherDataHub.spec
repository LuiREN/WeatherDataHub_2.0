# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.win32 import versioninfo

block_cipher = None

# Метаданные версии
version_info = versioninfo.VSVersionInfo(
    ffi=versioninfo.FixedFileInfo(
        filevers=(2, 0, 0, 0),
        prodvers=(2, 0, 0, 0),
        mask=0x3f,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0)
    ),
    kids=[
        versioninfo.StringFileInfo([
            versioninfo.StringTable("040904B0", [
                versioninfo.StringStruct("CompanyName", "WeatherDataHub"),
                versioninfo.StringStruct("FileDescription", "Weather Data Analysis Tool"),
                versioninfo.StringStruct("FileVersion", "2.0.0"),
                versioninfo.StringStruct("InternalName", "weatherdatahub"),
                versioninfo.StringStruct("OriginalFilename", "WeatherDataHub.exe"),
                versioninfo.StringStruct("ProductName", "WeatherDataHub"),
                versioninfo.StringStruct("ProductVersion", "2.0.0"),
            ])
        ]),
        versioninfo.VarFileInfo([versioninfo.VarStruct("Translation", [1033, 1200])])
    ]
)

# Анализ зависимостей
a = Analysis(
    ['main_window.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'requests',
        'bs4',
        'logging'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# Сжатие файлов
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# Создание EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WeatherDataHub',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Теперь можно отключить консоль
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)