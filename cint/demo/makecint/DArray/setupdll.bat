REM # makecint demonstration for Visual C++ 4.0
REM # DArray.cxx as Dynamic Link Library

move DArray.C DArray.cxx

REM # Create Makefile
makecint -mk Makefile -dl DArray.dll -H DArray.h -C++ DArray.cxx

REM # Compile
nmake /F Makefile CFG="DArray - Win32 Release" clean
nmake /F Makefile CFG="DArray - Win32 Release"
move Release\DArray.dll DArray.dll

REM # Test
cint DArray.cxx test.C
cint DArray.dll test.C


