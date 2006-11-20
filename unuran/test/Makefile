# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000


include $(ROOTSYS)/config/Makefile.config
include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------
CXXFLAGS += -g


ifeq ($(PLATFORM),win32)
EXTRALIBS        = "$(ROOTSYS)/lib/libMathMore.lib" "$(ROOTSYS)/lib/libMathCore.lib" "$(ROOTSYS)/lib/libUnuran.lib" "$(ROOTSYS)/lib/libMinuit.lib"
else
EXTRALIBS        =  -lUnuran -lMathCore -lMinuit
endif

#UNURANBASE=/Users/moneta/mathlibs/Unuran
#CXXFLAGS+=  -I$(UNURANBASE)/include 
#EXTRALIBS+=  $(UNURANBASE)/lib/libunuran.a 


#endif

TESTUNURANOBJ      = test_unuran.$(ObjSuf)
TESTUNURANSRC      = test_unuran.$(SrcSuf)
TESTUNURAN         = test_unuran$(ExeSuf)


UNURANSIMPLEOBJ      = unuranSimple.$(ObjSuf)
UNURANSIMPLESRC      = unuranSimple.$(SrcSuf)
UNURANSIMPLE         = unuranSimple$(ExeSuf)



UNURANDISTROBJ      = unuranDistr.$(ObjSuf)
UNURANDISTRSRC      = unuranDistr.$(SrcSuf)
UNURANDISTR         = unuranDistr$(ExeSuf)


UNURANMULTI2DOBJ      = unuranMulti2D.$(ObjSuf)
UNURANMULTI2DSRC      = unuranMulti2D.$(SrcSuf)
UNURANMULTI2D         = unuranMulti2D$(ExeSuf)

UNURANMULTI3DOBJ      = unuranMulti3D.$(ObjSuf)
UNURANMULTI3DSRC      = unuranMulti3D.$(SrcSuf)
UNURANMULTI3D         = unuranMulti3D$(ExeSuf)




OBJS          = $(UNURANDISTROBJ)  $(UNURANSIMPLEOBJ) $(UNURANMULTI2DOBJ) $(UNURANMULTI3DOBJ)


PROGRAMS      =  $(UNURANDISTR)  $(UNURANSIMPLE) $(UNURANMULTI2D) $(UNURANMULTI3D)

		  
.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)



# $(TESTUNURAN):  $(TESTUNURANOBJ)
# 		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
# 		@echo "$@ done"

$(UNURANDISTR): $(UNURANDISTROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(UNURANMULTI2D): $(UNURANMULTI2DOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(UNURANMULTI3D): $(UNURANMULTI3DOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



$(UNURANSIMPLE): $(UNURANSIMPLEOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
