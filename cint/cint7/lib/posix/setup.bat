makecint -mk Makefile -dl posix.dll -h winposix.h -C winposix.c -cint -Z0
nmake -f Makefile CFG="posix - Win32 Release"
del %cintsysdir%\include\posix.dll
move Release\posix.dll %cintsysdir%\include\posix.dll
nmake -f Makefile CFG="posix - Win32 Release" clean
echo off
rem echo #
rem echo #####################################
rem echo # Answer YES to following questions #
rem echo #####################################
rem echo #
rem del Release
rem rmdir Release
del Makefile
del G__*
del *.def
del make.bat
