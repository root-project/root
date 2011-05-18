REM # makecint demonstration for Visual C++ 4.0
REM # Complex.c as archived library

REM # Create Makefile
makecint -mk Makefile -o Complex -h Complex.h -C Complex.c -i stub.h

REM # Compile
nmake /F Makefile CFG="Complex - Win32 Release" clean
nmake /F Makefile CFG="Complex - Win32 Release"
move Release\Complex.exe Complex.exe

REM # Test
cint Complex.c stub.c test.c
Complex stub.c test.c


