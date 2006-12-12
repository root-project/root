REM # makecint demonstration for Symantec C++ 7.2
REM # Complex.cxx for archive library

move Complex.C Complex.cxx

REM # Create makefile and Complex.lnk
makecint -mk Makefile -o Complex -H Complex.h -C++ Complex.cxx

REM # Compile
smake -f Makefile clean
smake -f Makefile

REM # Test
cint Complex.cxx test.C
Complex test.C

