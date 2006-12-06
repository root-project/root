

rem #INC='-I/usr/X11R6/include'
rem #LIB='-L/usr/X11R6/lib -lGL -lGLU -lglut -lXmu -lX11 -lXi'
rem set LIB='-lGLAUX -lGLU32'

move ..\..\include/GL ..\..\include/GLx

makecint -mk Makefile -dl gl.dll %INC% -h TOP.h -l "c:\Program Files\Microsoft Visual Studio\VC98\Lib\glaux.lib" "c:\Program Files\Microsoft Visual Studio\VC98\Lib\glu32.lib" 

make 

move ..\..\include/GLx ..\..\include/GL

move gl.dll %CINTSYSDIR%\include\GL\gl.dll
make clean
del Makefile

