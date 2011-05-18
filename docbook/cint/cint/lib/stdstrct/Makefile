#
# lib/stdstrct/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/stdstrct.c
SRCCPPIFH     = $(CINTSYSDIR)/src/stdstrct.h
CPPIFC     = stdstrct.c
CPPIFH     = stdstrct.h

CPPHEADER  = \
		stdstr.h


$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -K -n$(CPPIFC) -NG__stdstrct $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-2 -K $(IPATH) $(MACRO) $(CPPHEADER)
