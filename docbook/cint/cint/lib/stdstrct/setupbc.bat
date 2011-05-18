echo > make.bat
echo > dmy.def
del make.bat
del *.def

echo > ..\..\include\stdfunc.dll
del ..\..\include\stdfunc.dll
makecint -mk Makestdfunc -dl stdfunc.dll -h stdfunc.h -cint -Z0
make.exe -f Makestdfunc 
move stdfunc.dll ..\..\include\stdfunc.dll
del G__*
del Makestdfunc

echo > ..\..\include\stdcxxfunc.dll
del ..\..\include\stdcxxfunc.dll
makecint -mk Makestdcxxfunc -dl stdcxxfunc.dll -H stdcxxfunc.h -cint -Z0
make.exe -f Makestdcxxfunc 
move stdcxxfunc.dll ..\..\include\stdcxxfunc.dll
del G__*
del Makestdcxxfunc
