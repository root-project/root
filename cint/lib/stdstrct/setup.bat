del ..\..\include\stdfunc.dll
makecint -mk Makestdfunc -dl stdfunc.dll -h stdfunc.h -cint -Z0
nmake -f Makestdfunc CFG="stdfunc - Win32 Release"
move Release\stdfunc.dll ..\..\include\stdfunc.dll
del G__*
del Makestdfunc
