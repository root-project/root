move %cintsysdir%\cint\include\windows.h %cintsysdir%\cint\include\_windows.h
move %cintsysdir%\cint\include\winsock.h %cintsysdir%\cint\include\_winsock.h

makecint -mk Makewin -dl win32api.dll -h +P cintwin.h -P winfunc.h -cint -Z0
make -f Makewin 
rem # CFG="win32api - Win32 Release"

del %cintsysdir%\cint5\include\win32api.dll
move Release\win32api.dll %cintsysdir%\cint5\include\win32api.dll
del win32api.lib
move Release\win32api.lib win32api.lib
move %cintsysdir%\cint\include\_windows.h %cintsysdir%\cint\include\windows.h
move %cintsysdir%\cint\include\_winsock.h %cintsysdir%\cint\include\winsock.h
echo off
rem echo #
rem echo #####################################
rem echo # Answer YES to following questions #
rem echo #####################################
rem echo #
rem del Release
rem rmdir Release
rem del G__*
del rem *.def
del makerem .bat

