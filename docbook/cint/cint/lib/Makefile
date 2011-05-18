

WIN32API =  ..\include\win32api.dll
POSIX    =  ..\include\posix.dll
STDFUNC  =  ..\include\stdfunc.dll
VECTOR   =  ..\stl\vector.dll


done : $(WIN32API) $(POSIX) $(STDFUNC) $(VECTOR)
	echo > done


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














