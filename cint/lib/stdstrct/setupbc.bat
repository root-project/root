del ..\..\include\stdfunc.dll
makecint -mk Makestdfunc -dl stdfunc.dll -h stdfunc.h -cint -Z0
make.exe -f Makestdfunc 
move stdfunc.dll ..\..\include\stdfunc.dll
del G__*
del Makestdfunc
