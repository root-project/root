
makecint -mk Make2 -dl stub.dll -H compiled.h -C++ compiled.cxx -i++ stub.h 
nmake -f Make2 CFG="stub - Win32 Debug"
move Debug\stub.dll stub.dll

cint main.cxx

cint complex.cxx stub.h main.cxx

del Debug
rmdir Debug
del G__*
del *.dll
