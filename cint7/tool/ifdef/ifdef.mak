##########################################################################
# Makefile for ifdef resolver
#
#  Author   : Masaharu Goto
#  Date     : 8 Feb 1994
#  Date     : 8 Jan 2001
#
##########################################################################
CPP = cl
CC  = cl
#-GX because of exception handling!
OPTIONS = -GX /nologo
OPTIMIZE = 

ifdef.exe: ifdef.obj get.obj winnt.obj
	$(CPP) $(OPTIMIZE) $(OPTIONS) -Foifdef.exe ifdef.obj get.obj winnt.obj

ifdef.obj: ifdef.cxx
	   $(CPP) $(OPTIMIZE) $(OPTIONS) -D_CRT_SECURE_NO_DEPRECATE -D_SECURE_SCL -DG__GET -c -Foifdef.obj ifdef.cxx

get.obj: get.c
	$(CC) $(OPTIMIZE) $(OPTIONS) -D_CRT_SECURE_NO_DEPRECATE -D_SECURE_SCL -DG__IFDEF -c -Foget.obj get.c

winnt.obj: ../../src/v6_winnt.cxx
	$(CPP) $(OPTIMIZE) $(OPTIONS) -c -D_CRT_SECURE_NO_DEPRECATE -D_SECURE_SCL -Fowinnt.obj ../../src/v6_winnt.cxx

clean:	
	del *.obj
#is moved !	del ifdef.exe
