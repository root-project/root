@echo off

set COUNTER=0
:loop
echo "Run number %COUNTER%"
testSetAddress.exe
if errorlevel 1 (
  echo "ERROR: The executable did not complete successfully!";
  exit /b 1;
)
set /a COUNTER+=1
if %COUNTER% LSS 33 goto loop
