
set QTDIR=c:\qt
set LIBS=%QTDIR%\lib\qt-mt230nc.lib %QTDIR%\lib\qtmain.lib 

moc lcdrange.h -o moc_lcdrange.cpp

makecint -mk Maketest7 -dl test7.dll -I %QTDIR%\include -I. -l %LIBS% -p -H test7.h -C++ lcdrange.cpp moc_lcdrange.cpp qtstatic.cxx

nmake -f Maketest7 CFG="test7 - Win32 Release"
move Release\test7.dll test7.dll

cint test7.cxx

del G__*
del moc_lcdrange.cpp
del *.obj
del *.def
del *.dll
del Maketest7
del make.bat
del Release\*
rmdir Release
