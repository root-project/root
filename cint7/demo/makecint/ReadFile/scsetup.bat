REM # makecint demonstration for Symantec C++ 7.2
REM # ReadFile.cxx as archived library

move ReadFile.C ReadFile.cxx

REM # Create Makefile
makecint -mk Makefile -o ReadFile -H ReadFile.h -C++ ReadFile.cxx


REM # Compile
smake clean
smake

REM # Test
cint ReadFile.cxx test.C
ReadFile test.C

