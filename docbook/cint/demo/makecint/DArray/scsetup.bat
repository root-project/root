REM # makecint demonstration for Symantec C++ 7.2
REM # DArray.cxx as archived library

move DArray.C DArray.cxx

REM # Create Makefile
makecint -mk Makefile -o DArray -H DArray.h -C++ DArray.cxx

REM # Compile
smake clean
smake

REM # Test
cint DArray.cxx test.C
DArray test.C


