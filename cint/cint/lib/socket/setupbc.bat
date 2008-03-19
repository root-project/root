def make.bat
del cintsock.def

bcc32 -emksockh.exe mksockh.c 
mksockh.exe
del mksockh.exe
del mksockh.obj
del mksockh.def

move %cintsysdir%\include\winsock.h %cintsysdir%\include\_winsock.h

makecint -mk Makefile -dl cintsock.dll -h cintsock.h -C cintsock.c -cint -Z0
rem -l "c:\Program Files\DevStudio\Vc\Lib\wsock32.lib"

make.exe -f Makefile 
echo > ..\..\include\cintsock.dll
del ..\..\include\cintsock.dll
move cintsock.dll %cintsysdir%\include\cintsock.dll
move %cintsysdir%\include\_winsock.h %cintsysdir%\include\winsock.h

make.exe -f Makefile clean
del Makefile
del G__*
del *.obj
del *.tds
