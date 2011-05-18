#
# lib/stlport4strm/Makefile
#
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

PLATFORM=stlport4strm

SRCCPPIFC     = $(CINTSYSDIR)/src/dict/$(PLATFORM).cxx
SRCCPPIFH     = $(CINTSYSDIR)/src/dict/$(PLATFORM).h
CPPIFC     = $(PLATFORM).cxx
CPPIFH     = $(PLATFORM).h

CPPHEADER  = iostrm.h fstrm.h sstrm.h linkdef.h

$(SRCCPPIFC) : $(CPPIFC)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -DG__GNUC=4 -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 

#	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -M0x10 -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 
