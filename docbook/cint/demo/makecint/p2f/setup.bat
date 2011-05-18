REM # makecint demonstration for Visual C++ 4.0
REM # p2f.cxx as archived library

REM # Right now, CINT ERTTI API classes are not exported from LIBCINT.DLL.
REM # So, this example does not work in Win32 environment. You can only
REM # try interpreted version and #pragma compile

move p2f.C p2f.cxx

REM # create makefile
REM # makecint -mk Makefile -o p2f -I%CINTSYSDIR% -I%CINTSYSDIR%\src -H p2f.h -C++ p2f.cxx

REM # compile object
REM # nmake /F Makefile CFG="p2f - Win32 Release" clean
REM # nmake /F Makefile CFG="p2f - Win32 Release"
REM # move Release\p2f.exe p2f.exe

REM # run test
cint p2f.cxx test.C
REM # p2f test.C