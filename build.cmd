@echo off
setlocal

if /I "%~1"=="go" goto go
if /I "%~1"=="clean" goto clean
if not "%~1"=="" (
    echo Usage: build.cmd [go^|clean]
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

echo Building native JSONL converter...
cl /nologo /TC /std:c11 /W4 /O2 /D_CRT_SECURE_NO_WARNINGS migrate.c /Fo:build\migrate.obj /Fe:build\migrate.exe
if errorlevel 1 exit /b 1
echo Building validation comparison tool...
cl /nologo /TC /std:c11 /W4 /O2 /D_CRT_SECURE_NO_WARNINGS compare_loss.c /Fo:build\compare_loss.obj /Fe:build\compare_loss.exe
if errorlevel 1 exit /b 1

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


:go
if not "%~2"=="" (
    echo Usage: build go
    exit /b 2
)
call "%~f0"
if errorlevel 1 exit /b 1
if not exist digit_approved_learning.jsonl (
    echo ERROR: Place the approved Digit JSONL snapshot in this repository as digit_approved_learning.jsonl
    exit /b 1
)
if not exist corpus_state_768000.stn (
    echo ERROR: Expected baseline checkpoint corpus_state_768000.stn
    exit /b 1
)
if not exist samples\synthetic_language.txt (
    echo ERROR: Synthetic test corpus missing.
    exit /b 1
)
if not exist runs mkdir runs
if errorlevel 1 exit /b 1
set "RUN=runs\go_%RANDOM%_%RANDOM%_%RANDOM%"
if exist "%RUN%" (
    echo ERROR: Run directory collision.
    exit /b 1
)
mkdir "%RUN%"
if errorlevel 1 exit /b 1
echo run_directory=%RUN% > "%RUN%\report.txt"
echo start_date=%DATE% %TIME% >> "%RUN%\report.txt"
echo Stage 1: Native self-test...
build\prepare.exe prepare samples\synthetic_language.txt "%RUN%\smoke_train.bin" "%RUN%\smoke_valid.bin" > "%RUN%\smoke_prepare.log" 2>&1
if errorlevel 1 goto failed
build\model.exe train "%RUN%\smoke_train.bin" "%RUN%\smoke_valid.bin" 200 "%RUN%\smoke.stn" > "%RUN%\smoke_train.log" 2>&1
if errorlevel 1 goto failed
build\model.exe eval "%RUN%\smoke.stn" "%RUN%\smoke_valid.bin" > "%RUN%\smoke_eval.log" 2>&1
if errorlevel 1 goto failed

echo Stage 2: Capturing input snapshot...
copy /b digit_approved_learning.jsonl "%RUN%\digit_snapshot.jsonl" >nul
if errorlevel 1 goto failed
echo Stage 3: Converting approved records...
build\migrate.exe convert "%RUN%\digit_snapshot.jsonl" "%RUN%\digit_corpus.txt" > "%RUN%\migration.log" 2>&1
if errorlevel 1 goto failed
echo Stage 4: Preparing train and validation...
build\prepare.exe prepare "%RUN%\digit_corpus.txt" "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" > "%RUN%\prepare.log" 2>&1
if errorlevel 1 goto failed

echo Stage 5: Evaluating baseline on NEW validation dataset...
build\model.exe eval corpus_state_768000.stn "%RUN%\digit_valid.bin" > "%RUN%\baseline_eval.log" 2>&1
if errorlevel 1 goto failed
set "BEST=corpus_state_768000.stn"
set "BESTLOG=%RUN%\baseline_eval.log"

echo Stage 6: Adapting weights to approved Digit snapshot...
build\model.exe adapt corpus_state_768000.stn "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" 64000 "%RUN%\state_64000.stn" > "%RUN%\train_64000.log" 2>&1
if errorlevel 1 goto failed
build\model.exe eval "%RUN%\state_64000.stn" "%RUN%\digit_valid.bin" > "%RUN%\eval_64000.log" 2>&1
if errorlevel 1 goto failed
call :choose "%RUN%\state_64000.stn" "%RUN%\eval_64000.log"
if errorlevel 1 goto failed

echo Stage 7: Resuming three bounded increments...
build\model.exe resume "%RUN%\state_64000.stn" "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" 64000 "%RUN%\state_128000.stn" > "%RUN%\train_128000.log" 2>&1
if errorlevel 1 goto failed
build\model.exe eval "%RUN%\state_128000.stn" "%RUN%\digit_valid.bin" > "%RUN%\eval_128000.log" 2>&1
if errorlevel 1 goto failed
call :choose "%RUN%\state_128000.stn" "%RUN%\eval_128000.log"
if errorlevel 1 goto failed

build\model.exe resume "%RUN%\state_128000.stn" "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" 64000 "%RUN%\state_192000.stn" > "%RUN%\train_192000.log" 2>&1
if errorlevel 1 goto failed
build\model.exe eval "%RUN%\state_192000.stn" "%RUN%\digit_valid.bin" > "%RUN%\eval_192000.log" 2>&1
if errorlevel 1 goto failed
call :choose "%RUN%\state_192000.stn" "%RUN%\eval_192000.log"
if errorlevel 1 goto failed

build\model.exe resume "%RUN%\state_192000.stn" "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" 64000 "%RUN%\state_256000.stn" > "%RUN%\train_256000.log" 2>&1
if errorlevel 1 goto failed
build\model.exe eval "%RUN%\state_256000.stn" "%RUN%\digit_valid.bin" > "%RUN%\eval_256000.log" 2>&1
if errorlevel 1 goto failed
call :choose "%RUN%\state_256000.stn" "%RUN%\eval_256000.log"
if errorlevel 1 goto failed

echo Stage 8: Capturing fixed-seed generation samples...
build\model.exe sample "%BEST%" "Question: Why does PHP seem hard? Answer:" 128 913 0.65 0 utf8 > "%RUN%\best_sample.hex" 2>&1
if errorlevel 1 goto failed
echo result=completed >> "%RUN%\report.txt"
echo best_checkpoint=%BEST% >> "%RUN%\report.txt"
echo end_date=%DATE% %TIME% >> "%RUN%\report.txt"
echo.
echo BUILD GO COMPLETE: %RUN%\report.txt
echo Best candidate: %BEST%
echo Validation comparisons are on the newly prepared holdout only.
echo Check sample output manually for legibility; coherence is not auto-qualified.
exit /b 0

:choose
build\compare_loss.exe "%BESTLOG%" "%~2" >> "%RUN%\report.txt" 2>&1
if errorlevel 2 exit /b 2
if not errorlevel 1 (
    set "BEST=%~1"
    set "BESTLOG=%~2"
)
exit /b 0

:failed
echo result=FAILED >> "%RUN%\report.txt"
echo failure_time=%DATE% %TIME% >> "%RUN%\report.txt"
echo.
echo BUILD GO FAILED. See logs and %RUN%\report.txt
exit /b 1
