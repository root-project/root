

WIN32API =  ..\include\win32api.dll
POSIX    =  ..\include\posix.dll
LONG     =  ..\include\long.dll
STDFUNC  =  ..\include\stdfunc.dll
VECTOR   =  ..\stl\vector.dll


done : $(WIN32API) $(POSIX) $(LONG) $(STDFUNC) $(VECTOR)
	echo > done


$(LONG) : longlong\longlong.h longlong\longdbl.h
	cd longlong
	setupbc.bat
	cd ..

$(STDFUNC) : stdstrct\stdfunc.h
	cd stdstrct
	setupbc.bat
	cd ..

$(VECTOR) : prec_stl\vector dll_stl\vec.h
	cd dll_stl
	setupbc.bat
	cd ..

$(WIN32API) : win32api\winfunc.h win32api\cintwin.h
	cd win32api
	setupbc.bat
	cd ..

$(POSIX) : posix\posix.h
	cd posix
	setupbc.bat
	cd ..














