REM # makecint demonstration for Visual C++ 4.0
REM # Complex.cxx as archived library

move Complex.C Complex.cxx

REM # Create makefile and Complex.lnk
makecint -mk Makefile -o Complex -H Complex.h -C++ Complex.cxx

REM # Compile
nmake /F Makefile CFG="Complex - Win32 Release" clean
nmake /F Makefile CFG="Complex - Win32 Release" 
move Release\Complex.exe Complex.exe

REM # Test
cint Complex.cxx test.C
Complex test.C

