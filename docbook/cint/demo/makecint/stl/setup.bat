makecint -mk Makefile -dl sample.dll -H sample.h -cint -M0x10
nmake -f Makefile CFG="sample - Win32 Release"
move Release\sample.dll sample.dll
cint test.cxx
del G__*
del release
