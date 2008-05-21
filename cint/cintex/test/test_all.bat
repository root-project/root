rem @echo off
set ROOTSYS=%cd%
set PYTHONDIR=%1\..

set PATH=%cd%\bin;%cd%\cint\cintex\test\dict;%cd%\cint\cintex\test\lib;%PATH%
set PYTHONPATH=%cd%\bin;%PYTHONPATH%

bin\root -b -q -l cint\cintex\test\test_Cintex.C
bin\root -b -q -l cint\cintex\test\test_Persistency.C
%PYTHONDIR%\python.exe cint\cintex\test\test_PyCintex_basics.py -b

