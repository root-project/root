makecint -mk Makefile -dl sample.dll -H sample.h -cint -M0x10
make.exe -f Makefile
cint test.cxx
del G__*
del Make*
del *.dll
