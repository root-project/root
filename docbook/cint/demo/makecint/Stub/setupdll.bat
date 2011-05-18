REM # makecint demonstration for Visual C++ 4.0
REM # Src.cxx as archived library

move Src.C Src.cxx

REM # Create Makefile
makecint -mk Makefile -dl Stub.dll -H Src.h -C++ Src.cxx -i++ Stub.h

REM # Compile
nmake /F Makefile CFG="Stub - Win32 Release" clean
nmake /F Makefile CFG="Stub - Win32 Release"
move Release\Stub.dll Stub.dll

REM # Test
cint Src.cxx Stub.C
cint Stub.dll Stub.C


