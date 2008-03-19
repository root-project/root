move %cintsysdir%\include\windows.h %cintsysdir%\include\_windows.h
move %cintsysdir%\include\winsock.h %cintsysdir%\include\_winsock.h

makecint -mk Makewin -dl win32api.dll -h +P cintwin.h -P winfunc.h -cint -Z0
smake -f Makewin 

del %cintsysdir%\include\win32api.dll
move win32api.dll %cintsysdir%\include\win32api.dll
del win32api.lib
move win32api.lib win32api.lib
move %cintsysdir%\include\_windows.h %cintsysdir%\include\windows.h
move %cintsysdir%\include\_winsock.h %cintsysdir%\include\winsock.h
echo off
echo #
echo #####################################
echo # Answer YES to following questions #
echo #####################################
echo #
rem del Release
rem rmdir Release
rem del G__*


