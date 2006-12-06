#
# lib/bcstream/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/bcstrm.cxx
SRCCPPIFH     = $(CINTSYSDIR)/src/bcstrm.h
CPPIFC     = bcstrm.cxx
CPPIFH     = bcstrm.h

CPPHEADER  = \
		iostrm.h \
		fstrm.h \
		stdiostr.h \
		strstrm.h 


$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER)
