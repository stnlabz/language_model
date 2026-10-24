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
echo Building original word-level GRU...
cl /nologo /TC /std:c11 /W4 /O2 /D_CRT_SECURE_NO_WARNINGS word_gru.c /Fo:build\word_gru.obj /Fe:build\word_gru.exe
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
rem Operator-directed: replace prior run artifacts with one stable run folder.
if exist "runs" (
    echo Clearing previous runs...
    rmdir /S /Q "runs"
    if exist "runs" (
        echo ERROR: Could not clear runs. No training started.
        exit /b 1
    )
)
mkdir "runs"
if errorlevel 1 exit /b 1
set "RUN=runs\gru_run"
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

echo Stage 5: Training word-level GRU on Digit's approved statements...
build\word_gru.exe train "%RUN%\digit_train.bin" "%RUN%\digit_valid.bin" "%RUN%\word_gru.stn" "%RUN%\best_sample.txt" > "%RUN%\word_gru.log" 2>&1
if errorlevel 1 goto failed
rem Qualification evidence must use the same decoder as production inference.
rem The training-generated best_sample.txt is trigram-constrained and cannot
rem establish how the checkpoint responds to prompts at runtime.
echo Stage 6: Capturing actual inference samples...
build\word_gru.exe sample "%RUN%\word_gru.stn" "Digit, explain yourself" 32 913 0.65 0 > "%RUN%\live_identity.txt" 2> "%RUN%\live_identity.err"
if errorlevel 1 goto failed
build\word_gru.exe sample "%RUN%\word_gru.stn" "What is your mission?" 32 913 0.65 0 > "%RUN%\live_mission.txt" 2> "%RUN%\live_mission.err"
if errorlevel 1 goto failed
build\word_gru.exe sample "%RUN%\word_gru.stn" "Hello Digit" 32 913 0.65 0 > "%RUN%\live_greeting.txt" 2> "%RUN%\live_greeting.err"
if errorlevel 1 goto failed
echo model=original_word_level_gru >> "%RUN%\report.txt"
echo checkpoint=%RUN%\word_gru.stn >> "%RUN%\report.txt"
echo sample_decoder=training_trigram_constrained >> "%RUN%\report.txt"
echo live_decoder=word_gru_sample >> "%RUN%\report.txt"
echo result=training_completed_language_unqualified >> "%RUN%\report.txt"
echo language_quality=NOT_VERIFIED >> "%RUN%\report.txt"
echo end_date=%DATE% %TIME% >> "%RUN%\report.txt"
echo.
echo WORD GRU TRAINING COMPLETE - LANGUAGE NOT QUALIFIED
echo ===== TRAINING-TIME CONSTRAINED SAMPLE =====
type "%RUN%\best_sample.txt"
if errorlevel 1 goto failed
echo ===== LIVE INFERENCE: SELF-EXPLANATION =====
type "%RUN%\live_identity.txt"
if errorlevel 1 goto failed
echo ===== LIVE INFERENCE: MISSION =====
type "%RUN%\live_mission.txt"
if errorlevel 1 goto failed
echo ===== LIVE INFERENCE: GREETING =====
type "%RUN%\live_greeting.txt"
if errorlevel 1 goto failed
echo =========================================
echo Model checkpoint: %RUN%\word_gru.stn
echo Training and validation logs: %RUN%\word_gru.log
echo WARNING: Training-time word-order constraints are NOT used by live inference.
echo WARNING: Inspect all live samples; English coherence is NOT verified by training.
exit /b 0

:failed
echo result=FAILED >> "%RUN%\report.txt"
echo failure_time=%DATE% %TIME% >> "%RUN%\report.txt"
echo.
echo BUILD GO STOPPED. See logs and %RUN%\report.txt
exit /b 1
