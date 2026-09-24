@echo off
setlocal

if "%~1"=="" (
  echo Usage: run_validation.bat C:\path\to\new_run_scalar.txt
  exit /b 2
)

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%validation_output" rmdir /s /q "%SCRIPT_DIR%validation_output"

python "%SCRIPT_DIR%apply_foundation_environment_v11.py" ^
  --model "%SCRIPT_DIR%model\candidate_v11.joblib" ^
  --input "%~1" ^
  --output "%SCRIPT_DIR%validation_output"

if errorlevel 1 exit /b %errorlevel%
echo Validation finished: %SCRIPT_DIR%validation_output
