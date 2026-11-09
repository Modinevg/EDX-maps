@echo off
rem ===========================================================================
rem  EDX processing - one-time setup of a private Python environment.
rem
rem  Downloads Python 3.12 (official python.org build, package from nuget.org)
rem  and installs the libraries of requirements.txt into it. Nothing is
rem  installed system-wide: no admin rights, no registry entries, PATH and
rem  other Python installations untouched.
rem    Location:  %LOCALAPPDATA%\EDX_processing\python  (or set EDX_ENV)
rem    Uninstall: delete that folder.
rem  Running it again brings the libraries to the versions of requirements.txt.
rem  Afterwards start the program with run_edx_gui.bat.
rem ===========================================================================
setlocal
set "PYVER=3.12.10"
if not defined EDX_ENV set "EDX_ENV=%LOCALAPPDATA%\EDX_processing"
set "PYDIR=%EDX_ENV%\python"
set "PY=%PYDIR%\tools\python.exe"
set "REQ=%~dp0requirements.txt"
rem isolate the environment from other Python installations and user packages
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="

echo EDX processing - environment setup
echo   Python %PYVER% and libraries in "%PYDIR%"
echo.
if not exist "%REQ%" goto no_req

rem an existing Python is reused only if its standard library is complete (Lib\os.py is the landmark
rem by which Python finds it; os itself is built in) and pip works
set "CHECK=import os, sys, json, ssl, pip; sys.exit(not os.path.isfile(os.path.join(sys.prefix, 'Lib', 'os.py')))"
"%PY%" -c "%CHECK%" >nul 2>nul
if not errorlevel 1 goto have_python

rem ---- download and unpack Python ----
where curl.exe >nul 2>nul
if errorlevel 1 goto no_tools
where tar.exe >nul 2>nul
if errorlevel 1 goto no_tools
if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
if not exist "%EDX_ENV%" mkdir "%EDX_ENV%"
echo Downloading Python %PYVER% (15 MB)...
curl.exe -fL --retry 3 -o "%EDX_ENV%\python-%PYVER%.zip" "https://api.nuget.org/v3-flatcontainer/python/%PYVER%/python.%PYVER%.nupkg"
if errorlevel 1 goto net_fail
mkdir "%PYDIR%"
rem -m: files get the current date (with the 2025 dates of the archive, cleaners of old files may remove them)
tar.exe -xmf "%EDX_ENV%\python-%PYVER%.zip" -C "%PYDIR%" tools
if errorlevel 1 goto fail
del "%EDX_ENV%\python-%PYVER%.zip"
"%PY%" -c "%CHECK%" >nul 2>nul
if errorlevel 1 goto fail

:have_python
"%PY%" -c "import sys; print('Python', sys.version.split()[0], 'OK')"
echo.
echo Installing libraries (about 150 MB to download on the first run)...
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r "%REQ%"
if errorlevel 1 goto net_fail

echo.
echo Checking the program modules...
pushd "%~dp0"
"%PY%" -c "import edx_core, edx_gui, matplotlib, PIL, tifffile, pptx, numpy, scipy, pyqtgraph; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pyqtgraph', pyqtgraph.__version__, '| Qt binding', pyqtgraph.Qt.QT_LIB)"
set "ERR=%errorlevel%"
popd
if not "%ERR%"=="0" goto fail

echo.
echo Done. Start the program with run_edx_gui.bat
if not defined EDX_NOPAUSE pause
exit /b 0

:no_req
echo ERROR: requirements.txt not found next to setup_edx_env.bat.
goto fail
:no_tools
echo ERROR: curl.exe or tar.exe not found (they are part of Windows 10 1803 and later).
goto fail
:net_fail
echo ERROR: download failed - check the internet connection (proxy?) and run this file again.
:fail
echo.
echo Setup FAILED.
if not defined EDX_NOPAUSE pause
exit /b 1
