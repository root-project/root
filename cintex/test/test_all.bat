rem @echo off
set ROOTSYS=%cd%
set PYTHONDIR=%1\..

set PATH=%cd%\bin;%PATH%
set PYTHONPATH=%cd%\bin;%PYTHONPATH%

bin\root -b -q -l cintex\test\test_Cintex.C
bin\root -b -q -l cintex\test\test_Persistency.C
%PYTHONDIR%\python.exe cintex\test\test_PyCintex_basics.py

