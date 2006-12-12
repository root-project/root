REM # makecint demonstration for Symantec C++ 7.2
REM # Src.cxx as Dynamic Link Library

move Src.C Src.cxx

REM # Create Makefile
makecint -mk Makefile -dl Stub.dll -H Src.h -C++ Src.cxx -i++ Stub.h

REM # Compile
smake clean
smake

REM # Test
cint Src.cxx Stub.C
cint Stub.dll Stub.C

