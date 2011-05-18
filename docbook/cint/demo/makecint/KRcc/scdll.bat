REM # makecint demonstration for Symantec C++ 7.2
REM # Complex.c as Dynamic Link Library

REM # Create Makefile
makecint -mk Makefile -dl Complex.dll -h Complex.h -C Complex.c -i stub.h

REM # Compile
smake clean
smake

REM # Test
cint Complex.c stub.c test.c
cint Complex.dll stub.c test.c


