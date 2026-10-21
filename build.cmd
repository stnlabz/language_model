@echo off
setlocal

if /I "%~1"=="clean" goto clean
if not "%~1"=="" (
    echo Usage: build.cmd [clean]
    exit /b 2
)

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe was not found. Run from an MSVC Developer Command Prompt or initialize the MSVC build environment.
    exit /b 1
)

if not exist build mkdir build
if errorlevel 1 (
    echo ERROR: Could not create build directory.
    exit /b 1
)

echo Building STN-Labz GRU model...
cl /nologo /TC /std:c11 /W4 /O2 /D_CRT_SECURE_NO_WARNINGS model.c /Fo:build\model.obj /Fe:build\model.exe
if errorlevel 1 (
    echo MODEL BUILD FAILED.
    exit /b 1
)

echo Building STN-Labz corpus preparer...
cl /nologo /TC /std:c11 /W4 /O2 /D_CRT_SECURE_NO_WARNINGS prepare.c /Fo:build\prepare.obj /Fe:build\prepare.exe
if errorlevel 1 (
    echo PREPARE BUILD FAILED.
    exit /b 1
)

if not exist build\model.exe (
    echo MODEL OUTPUT MISSING.
    exit /b 1
)
if not exist build\prepare.exe (
    echo PREPARE OUTPUT MISSING.
    exit /b 1
)

echo.
echo BUILD GREEN: build\model.exe and build\prepare.exe
echo NOTE: Build completed; Windows runtime and checkpoint compatibility tests have not run.
exit /b 0

:clean
echo Cleaning STN-Labz GRU build...
if exist build rmdir /S /Q build
if exist build (
    echo CLEAN FAILED.
    exit /b 1
)
echo CLEAN GREEN.
exit /b 0
