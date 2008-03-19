REM # makecint demonstration for Symantec C++ 7.2
REM # Src.cxx as archived library

move Src.C Src.cxx

REM # Create Makefile
makecint -mk Makefile -o Stub -H Src.h -C++ Src.cxx -i++ Stub.h

REM # Compile
smake clean
smake

REM # Test
cint Src.cxx Stub.C
Stub Stub.C


