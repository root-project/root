#
# lib/stream/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/libstrm.cxx
SRCCPPIFH     = $(CINTSYSDIR)/src/libstrm.h
CPPIFC     = libstrm.cxx
CPPIFH     = libstrm.h

CPPHEADER  = \
		iostrm.h \
		fstrm.h \
		stdiostr.h \
		strm.h \
		strstrm.h 


$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER)
