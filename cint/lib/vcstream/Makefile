#
# lib/vcstream/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/vcstrm.cxx
SRCCPPIFH     = $(CINTSYSDIR)/src/vcstrm.h
CPPIFC     = vcstrm.cxx
CPPIFH     = vcstrm.h

CPPHEADER  = \
		iostrm.h \
		fstrm.h \
		stdiostr.h \
		strstrm.h 


$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER)
