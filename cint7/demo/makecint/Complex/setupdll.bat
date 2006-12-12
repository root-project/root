REM # makecint demonstration for Visual C++ 4.0
REM # Complex.cxx as Dynamic Link Library

move Complex.C Complex.cxx

REM # Create makefile and Complex.lnk
makecint -mk Makefile -dl Complex.dll -H Complex.h -C++ Complex.cxx

REM # Compile
nmake /F Makefile CFG="Complex - Win32 Release" clean
nmake /F Makefile CFG="Complex - Win32 Release" 
move Release\Complex.dll Complex.dll

REM # Test
cint Complex.cxx test.C
cint Complex.dll test.C

