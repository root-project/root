
set QTDIR=c:\qt

rem /D QT_DLL /D QT_THREAD_SUPPORT

makecint -mk Makefile -dl qtcint.dll -H +P qtcint.h qtclasses.h qtglobals.h qtfunctions.h -P qtdummy.h -C++ qtstatic.cxx -I%QTDIR%\include -I. -l %QTDIR%\lib\qt-mt230nc.lib %QTDIR%\lib\qtmain.lib %QTDIR%\lib\qutil.lib -cint -Z0

nmake -f Makefile CFG="qtcint - Win32 Release" clean

nmake -f Makefile CFG="qtcint - Win32 Release"
move Release\qtcint.dll \cint\include\qtcint.dll

del *.def
del G__cpp_qtcint.*
del Makefile
del make.bat
del Release\*
rmdir Release
