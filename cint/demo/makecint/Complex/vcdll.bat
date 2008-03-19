
REM # Visual C++ DLL test

move Complex.C Complex.cxx

REM # Create makefile and Complex.lnk
makecint -mk Makefile -dl Complex.dll -H Complex.h -C++ Complex.cxx

REM # Compile
nmake /f Makefile CFG="Complex - Win32 Release" clean
nmake /f Makefile CFG="Complex - Win32 Release" 
del Complex.dll
move Release\Complex.dll Complex.dll

REM # Test
cint Complex.cxx test.C
cint Complex.dll  test.C

