makecint -mk Makefile -m -Ic:/cint -o UserMain -H UserMain.h -C++ UserMain.cxx
make.exe -f Makefile
UserMain
make.exe -f Makefile clean
del Makefile
del UserMain.tds
del UserMain.exe
del UserMain.obj
del G__*
