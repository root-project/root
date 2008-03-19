
REM # Visual C++ EXE test

move Complex.C Complex.cxx

REM # Create makefile
makecint -mk Makefile -o Complex -H Complex.h -C++ Complex.cxx

REM # Compile
nmake /f Makefile CFG="Complex - Win32 Release" clean
nmake /f Makefile CFG="Complex - Win32 Release" 
del Complex.exe
move Release\Complex.exe Complex.exe

REM # Test
cint Complex.cxx test.C
Complex test.C

