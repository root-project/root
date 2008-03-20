#
# lib/snstream/Makefile
#
CP         = cp
MV	   = mv

#CINTSYSDIR = ../../..

SRCCPPIFC     = ../../src/dict/iccstrm.cxx
SRCCPPIFH     = ../../src/dict/iccstrm.h
CPPIFC     = iccstrm.cxx
CPPIFH     = iccstrm.h

CPPHEADER  = iostrm.h fstrm.h sstrm.h linkdef.h

#		stdiostr.h \


$(SRCCPPIFC) : $(CPPIFC)
	cint_tmp ../sunstrm/length.cxx 255 $(CPPIFC) $(CPPIFH)
	$(MV) $(CPPIFC) $(SRCCPPIFC)		
	$(MV) $(CPPIFH) $(SRCCPPIFH)		


$(CPPIFC) : $(CPPHEADER)
	cint_tmp -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 

#	cint -Z0 -n$(CPPIFC) -NG__stream $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT -M0x10 -c-1 $(IPATH) $(MACRO) $(CPPHEADER) 
