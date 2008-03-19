makecint -mk Makeit -dl statistics.dll -c statistics.c
nmake -f Makeit CFG="statistics - Win32 Release"
move Release\\statistics.dll statistics.dl

makecint -mk Makeit  -dl array.dll -c array.c -DG__WIN32
nmake -f Makeit CFG="array - Win32 Release"
move Release\\array.dll array.dl

makecint -mk Makeit  -dl carray.dll -c carray.c
nmake -f Makeit CFG="carray - Win32 Release"
move Release\\carray.dll carray.dl

makecint -mk Makeit  -dl fft.dll -c fft.c
nmake -f Makeit CFG="fft - Win32 Release"
move Release\\fft.dll fft.dl

makecint -mk Makeit  -dl lsm.dll -c lsm.c
nmake -f Makeit CFG="lsm - Win32 Release"
move Release\\lsm.dll lsm.dl

makecint -mk Makeit  -dl xgraph.dll -c xgraph.c
nmake -f Makeit CFG="xgraph - Win32 Release"
move Release\\xgraph.dll xgraph.dl

del ReadF.C
makecint -mk Makeit  -dl ReadF.dl -H ReadF.h -C++ ReadF.cxx
nmake -f Makeit CFG="ReadF - Win32 Release"
move Release\\ReadF.dll ReadF.dl

copy iosenum.win32 iosenum.h

rem makecint -mk Makeit  -dl RegE.dl -H RegE.h -C++ RegE.cxx
rem nmake -f Makeit CFG="RegE - Win32 Release"
rem move Release\\RegE.dll RegE.dl

del G__*
del *.def
del Release\*.obj
del Release\*.lib
del Release\*.exp
del Release\*.pch
rmdir Release


