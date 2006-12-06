makecint -mk Makefile -dl long.dll -H longlong.h longdbl.h -DG__BORLAND -cint -Z0
make.exe -f Makefile 
echo > %cintsysdir%\include\long.dll
del %cintsysdir%\include\long.dll
move long.dll %cintsysdir%\include\long.dll

make.exe -f Makefile clean
del Makefile
del G__*
del *.obj
del *.tds
