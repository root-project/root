#
# lib/cbstream/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/cbstrm.cpp
SRCCPPIFH     = $(CINTSYSDIR)/src/cbstrm.h
CPPIFC     = cbstrm.cpp
CPPIFH     = cbstrm.h

CPPHEADER  = iostrm.h fstrm.h strstrm.h linkdef.h

#		stdiostr.h \


$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 

#	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -M0x10 -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 
