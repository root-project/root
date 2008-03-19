
set QTDIR=c:\qt
set LIBS=%QTDIR%\lib\qt-mt230nc.lib %QTDIR%\lib\qtmain.lib 

moc lcdrange.h -o moc_lcdrange.cpp
moc cannon.h -o moc_cannon.cpp

makecint -mk Maketest8 -dl test8.dll -DTEST8 -I %QTDIR%\include -I. -l %LIBS% -p -H test8.h -C++ lcdrange.cpp cannon.cpp moc_lcdrange.cpp moc_cannon.cpp qtstatic.cxx

nmake -f Maketest8 CFG="test8 - Win32 Release"
move Release\test8.dll test8.dll

cint test8.cxx

del G__*
del moc_lcdrange.cpp
del *.obj
del *.def
del *.dll
del Maketest8
del make.bat
del Release\*
rmdir Release
