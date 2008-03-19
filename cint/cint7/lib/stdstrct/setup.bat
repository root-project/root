del ..\..\include\stdfunc.dll
makecint -mk Makestdfunc -dl stdfunc.dll -h stdfunc.h -cint -Z0
nmake -f Makestdfunc CFG="stdfunc - Win32 Release"
move Release\stdfunc.dll ..\..\include\stdfunc.dll
del G__*
del Makestdfunc

del ..\..\include\stdcxxfunc.dll
makecint -mk Makestdcxxfunc -dl stdcxxfunc.dll -H stdcxxfunc.h -cint -Z0
nmake -f Makestdcxxfunc CFG="stdcxxfunc - Win32 Release"
move Release\stdcxxfunc.dll ..\..\include\stdcxxfunc.dll
del G__*
del Makestdcxxfunc
del *.def
del make.bat
