@echo off
rem EDX spectrum-image processing GUI (FEI TIA .emi/.ser, EDAX .spd/.lsd/.spc)
rem Uses the private Python environment made by setup_edx_env.bat if it exists,
rem otherwise python from PATH (with numpy, scipy, PyQt5, pyqtgraph, ... installed).
setlocal
if not defined EDX_ENV set "EDX_ENV=%LOCALAPPDATA%\EDX_processing"
set "PY=%EDX_ENV%\python\tools\python.exe"
if not exist "%PY%" goto system_python
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
goto run
:system_python
set "PY=python"
:run
"%PY%" "%~dp0edx_gui.py" %*
if errorlevel 1 pause
