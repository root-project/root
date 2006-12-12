REM # makecint demonstration for Visual C++ 4.0
REM # ReadFile.cxx as Dynamic Link Library

move ReadFile.C ReadFile.cxx

REM # Create Makefile
makecint -mk Makefile -dl ReadFile.dll -H ReadFile.h -C++ ReadFile.cxx


REM # Compile
nmake /F Makefile CFG="ReadFile - Win32 Release" clean
nmake /F Makefile CFG="ReadFile - Win32 Release" 
move Release\ReadFile.dll ReadFile.dll

REM # Test
cint ReadFile.cxx test.C
cint ReadFile.dll test.C

