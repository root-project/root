REM # makecint demonstration for Symantec C++ 7.2
REM # DArray.cxx as Dynamic Link Library

move DArray.C DArray.cxx

REM # Create Makefile
makecint -mk Makefile -dl DArray.dll -H DArray.h -C++ DArray.cxx

REM # Compile
smake clean
smake

REM # Test
cint DArray.cxx test.C
cint DArray.dll test.C


