import os
import sys
import winshell
from win32com.client import Dispatch

def create_shortcut():
    desktop = winshell.desktop()
    path = os.path.join(desktop, "WeatherDataHub.lnk")
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    
    # Путь к EXE
    shortcut.Targetpath = os.path.abspath("WeatherDataHub.exe")
    shortcut.WorkingDirectory = os.path.abspath(".")
    shortcut.IconLocation = os.path.abspath("icon.ico")
    shortcut.save()

if __name__ == '__main__':
    create_shortcut()
