REM # makecint demonstration for Visual C++ 4.0
REM # Complex.c as Dynamic Link Library

REM # Create Makefile
makecint -mk Makefile -dl Complex.dll -h Complex.h -C Complex.c -i stub.h

REM # Compile
nmake /F Makefile CFG="Complex - Win32 Release" clean
nmake /F Makefile CFG="Complex - Win32 Release"
move Release\Complex.dll Complex.dll

REM # Test
cint Complex.c stub.c test.c
cint Complex.dll stub.c test.c


