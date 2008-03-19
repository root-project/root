makecint -mk Makefile -m -I%CINTSYSDIR% -o UserMain -H UserMain.h -C++ UserMain.cxx
nmake -f Makefile CFG="UserMain - Win32 Release"
move Release\UserMain.exe UserMain.exe
UserMain
del G__*
del Release
rmdir Release
del Makefile
