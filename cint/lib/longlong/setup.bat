makecint -mk Makefile -dl long.dll -H longlong.h longdbl.h -cint -Z0
nmake -f Makefile CFG="long - Win32 Release"
del %cintsysdir%\include\long.dll
move Release\long.dll %cintsysdir%\include\long.dll
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
