@echo off
if "%~1"=="" (
  echo Usage: run_validation.bat NEW_SCALAR.txt
  exit /b 2
)
python validate_new_data.py --input "%~1" --model model\candidate_v8.joblib --output-dir validation_output
