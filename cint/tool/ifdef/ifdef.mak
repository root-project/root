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

OPTIMIZE = 

ifdef.exe : ifdef.obj get.obj winnt.obj
	$(CPP) $(OPTIMIZE) -o ifdef.exe ifdef.obj get.obj winnt.obj

ifdef.obj: ifdef.cxx
	$(CPP) $(OPTIMIZE) -DG__GET -c -o ifdef.obj ifdef.cxx

get.obj: get.c
	$(CC) $(OPTIMIZE) -DG__IFDEF -c -o get.obj get.c

winnt.obj: ../../src/winnt.c
	$(CC) $(OPTIMIZE) -c -o winnt.obj ../../src/winnt.c


clean:
	del *.obj
	del ifdef.exe
