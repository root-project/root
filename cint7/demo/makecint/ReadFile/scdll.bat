REM # makecint demonstration for Symantec C++ 7.2
REM # ReadFile.cxx as Dynamic Link Library

move ReadFile.C ReadFile.cxx

REM # Create Makefile
makecint -mk Makefile -dl ReadFile.dll -H ReadFile.h -C++ ReadFile.cxx


REM # Compile
smake clean
smake

REM # Test
cint ReadFile.cxx test.C
cint ReadFile.dll test.C

