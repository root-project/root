
set QTDIR=c:\qt
set LIBS=%QTDIR%\lib\qt-mt230nc.lib %QTDIR%\lib\qtmain.lib 

moc lcdrange.h -o moc_lcdrange.cpp
moc cannon.h -o moc_cannon.cpp

makecint -mk Maketest9 -dl test9.dll -DTEST8 -DTEST9 -I %QTDIR%\include -I. -l %LIBS% -p -H test8.h -C++ lcdrange.cpp cannon.cpp moc_lcdrange.cpp moc_cannon.cpp qtstatic.cxx

nmake -f Maketest9 CFG="test9 - Win32 Release"
move Release\test9.dll test9.dll

cint test9.cxx

del G__*
del moc_lcdrange.cpp
del *.obj
del *.def
del *.dll
del Maketest9
del make.bat
del Release\*
rmdir Release
