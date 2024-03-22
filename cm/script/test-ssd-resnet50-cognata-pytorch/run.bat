@echo off

echo =======================================================

set CUR_DIR=%cd%

rem rem Patch model
rem cd %CM_ABTF_SSD_PYTORCH%
rem if not exist patchfile-20240308.patch (
rem   echo.
rem   echo Patching ABTF SRC
rem   echo.
rem 
rem   copy %CM_TMP_CURRENT_SCRIPT_PATH%\patches\cognata\patchfile-20240308.patch .
rem   patch -s -p0 < patchfile-20240308.patch
rem   IF %ERRORLEVEL% NEQ 0 EXIT %ERRORLEVEL%
rem )

cd %CUR_DIR%

echo.
%CM_PYTHON_BIN_WITH_PATH% %CM_ABTF_SSD_PYTORCH%\test_image.py --pretrained-model "%CM_ML_MODEL_FILE_WITH_PATH%" --dataset %CM_ABTF_DATASET% --config %CM_ABTF_ML_MODEL_CONFIG% --input %CM_INPUT_IMAGE% --output %CM_OUTPUT_IMAGE%
IF %ERRORLEVEL% NEQ 0 EXIT %ERRORLEVEL%
