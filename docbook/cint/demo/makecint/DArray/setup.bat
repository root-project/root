REM # makecint demonstration for Visual C++ 4.0
REM # DArray.cxx as archived library

move DArray.C DArray.cxx

REM # Create Makefile
makecint -mk Makefile -o DArray -H DArray.h -C++ DArray.cxx

REM # Compile
nmake /F Makefile CFG="DArray - Win32 Release" clean
nmake /F Makefile CFG="DArray - Win32 Release"
move Release\DArray.exe DArray.exe

REM # Test
cint DArray.cxx test.C
DArray test.C


