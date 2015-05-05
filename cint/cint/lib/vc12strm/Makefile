#
# lib/snstream/Makefile
#
#CP         = copy
#MV	   = move
CP         = cp
MV	   = mv

CINTSYSDIR = ../..

SRCCPPIFC     = $(CINTSYSDIR)/src/vc7strm.cxx
SRCCPPIFH     = $(CINTSYSDIR)/src/vc7strm.h
CPPIFC     = vc7strm.cxx
CPPIFH     = vc7strm.h

CPPHEADER  = iostrm.h fstrm.h sstrm.h linkdef.h

#		stdiostr.h \


$(SRCCPPIFC) : $(CPPIFC)
	cint ../snstream/length.cxx 255 $(CPPIFC) $(CPPIFH)
#	$(MV) $(CPPIFC) $(SRCCPPIFC)		
#	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 

#	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -M0x10 -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 
