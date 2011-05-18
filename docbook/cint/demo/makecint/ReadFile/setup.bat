REM # makecint demonstration for Visual C++ 4.0
REM # ReadFile.cxx as archived library

move ReadFile.C ReadFile.cxx

REM # Create Makefile
makecint -mk Makefile -o ReadFile -H ReadFile.h -C++ ReadFile.cxx


REM # Compile
nmake /F Makefile CFG="ReadFile - Win32 Release" clean
nmake /F Makefile CFG="ReadFile - Win32 Release" 
move Release\ReadFile.exe ReadFile.exe

REM # Test
cint ReadFile.cxx test.C
ReadFile test.C

