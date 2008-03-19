REM # makecint demonstration for Symantec C++ 7.2
REM # Complex.c as archived library

REM # Create Makefile
makecint -mk Makefile -o Complex -h Complex.h -C Complex.c -i stub.h

REM # Compile
smake clean
smake

REM # Test
cint Complex.c stub.c test.c
Complex stub.c test.c


