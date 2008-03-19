

makecint -mk Makefile -dl mt.dll -h mt.h -C mt.c
nmake -f Makefile CFG="mt - Win32 Release"
move Release\mt.dll mt.dll

cint main.cxx

rem del mt.dll
rem del Release
rem rmdir Release
