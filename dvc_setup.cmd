@echo off
REM =============================================================================
REM DVC Setup Script for AutoMend Pipeline (Windows CMD Version)
REM =============================================================================

REM -----------------------------------------------------------------------------
REM Step 0: Ensure Git is initialized (DVC requires Git)
REM -----------------------------------------------------------------------------
IF NOT EXIST ".git" (
    echo Initializing Git repository...
    git init
)

REM -----------------------------------------------------------------------------
REM Step 1: Initialize DVC
REM -----------------------------------------------------------------------------
echo Initializing DVC...
dvc init

REM -----------------------------------------------------------------------------
REM Step 2: Create DVC remote storage (Local)
REM -----------------------------------------------------------------------------
echo Setting up local DVC remote...

IF NOT EXIST "..\dvc-storage" (
    mkdir ..\dvc-storage
)

dvc remote add -d localremote ..\dvc-storage

REM -----------------------------------------------------------------------------
REM Step 3: Track source data files
REM -----------------------------------------------------------------------------
echo Tracking source data files...

IF EXIST "data\external\Stack_Qns_pl.csv" (
    dvc add data\external\Stack_Qns_pl.csv
    echo   - Tracked Stack_Qns_pl.csv
)

IF EXIST "data\external\Stack_Ans_pl.csv" (
    dvc add data\external\Stack_Ans_pl.csv
    echo   - Tracked Stack_Ans_pl.csv
)

REM -----------------------------------------------------------------------------
REM Step 4: Create data\.gitignore
REM -----------------------------------------------------------------------------
echo Creating data\.gitignore...

IF NOT EXIST data mkdir data

(
echo # Ignore data files (tracked by DVC)
echo raw/
echo processed/
echo validated/
echo training/
echo.
echo # Keep external directory structure
echo !external/
echo external/*
echo !external/.gitkeep
echo !external/*.dvc
) > data\.gitignore

REM -----------------------------------------------------------------------------
REM Step 5: Create directory structure + .gitkeep files
REM -----------------------------------------------------------------------------
IF NOT EXIST data\external mkdir data\external
IF NOT EXIST data\raw mkdir data\raw
IF NOT EXIST data\processed mkdir data\processed
IF NOT EXIST data\validated mkdir data\validated
IF NOT EXIST data\training mkdir data\training
IF NOT EXIST logs mkdir logs
IF NOT EXIST reports\validation mkdir reports\validation
IF NOT EXIST reports\statistics mkdir reports\statistics
IF NOT EXIST reports\bias mkdir reports\bias

type nul > data\external\.gitkeep
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > data\validated\.gitkeep
type nul > data\training\.gitkeep
type nul > logs\.gitkeep
type nul > reports\validation\.gitkeep
type nul > reports\statistics\.gitkeep
type nul > reports\bias\.gitkeep

REM -----------------------------------------------------------------------------
REM Step 6: Add DVC files to Git
REM -----------------------------------------------------------------------------
echo Adding DVC files to Git...

git add dvc.yaml dvc.lock params.yaml .dvc .dvcignore
git add data\external\*.dvc data\.gitignore
git add data\*\*.gitkeep logs\.gitkeep reports\*\*.gitkeep

REM -----------------------------------------------------------------------------
REM Step 7: Verify Setup
REM -----------------------------------------------------------------------------
echo.
echo ==============================================
echo DVC Setup Complete!
echo ==============================================
echo.

echo Configured remote:
dvc remote list
echo.

echo Tracked files:
dvc status
echo.

echo Next steps:
echo   1. Commit changes:
echo        git commit -m "Initialize DVC tracking"
echo   2. Run pipeline:
echo        dvc repro
echo   3. Push data:
echo        dvc push
echo.

pause