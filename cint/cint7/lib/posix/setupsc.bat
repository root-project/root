makecint -mk Makefile -dl posix.dll -h winposix.h -C winposix.c -cint -Z0
smake -f Makefile 
del %cintsysdir%\include\posix.dll
move posix.dll %cintsysdir%\include\posix.dll
smake -f Makefile clean
echo off
echo #
echo #####################################
echo # Answer YES to following questions #
echo #####################################
echo #
del *.obj
del Makefile
del G__*
