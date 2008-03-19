REM # makecint demonstration for Symantec C++
REM # Complex.cxx as Dymanic Link Library

move Complex.C Complex.cxx

REM # Create makefile and Complex.lnk
makecint -mk Makefile -dl Complex.dll -H Complex.h -C++ Complex.cxx

REM # Compile
smake -f Makefile clean
smake -f Makefile

REM # Test
cint Complex.cxx test.C
cint Complex.dll  test.C

