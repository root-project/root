move %cintsysdir%\include\windows.h %cintsysdir%\include\_windows.h
move %cintsysdir%\include\winsock.h %cintsysdir%\include\_winsock.h
del make.bat
del win32api.def

makecint -mk Makewin -dl win32api.dll -h +P cintwin.h -P winfunc.h -cint -Z0
make.exe -f Makewin 

echo > %cintsysdir%\include\win32api.dll
del %cintsysdir%\include\win32api.dll
move win32api.dll %cintsysdir%\include\win32api.dll
echo > win32api.lib
del win32api.lib
move win32api.lib win32api.lib
move %cintsysdir%\include\_windows.h %cintsysdir%\include\windows.h
move %cintsysdir%\include\_winsock.h %cintsysdir%\include\winsock.h

rem make.exe -f Makewin clean
del Makewin
del G__*
del *.obj
del win32api.tds

