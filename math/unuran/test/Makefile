# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../..
include $(ROOTSYS)/config/Makefile.config
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------


ifeq ($(PLATFORM),win32)
EXTRALIBS        = "$(ROOTSYS)/lib/libMathCore.lib" "$(ROOTSYS)/lib/libUnuran.lib" "$(ROOTSYS)/lib/libMinuit.lib"
else
CXXFLAGS       += -g 
EXTRALIBS      =  -lUnuran -lMathCore -lMinuit
endif


ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
ifeq ($(PLATFORM),win32)
EXTRALIBS += "$(CLHEPBASE)/lib/CLHEP.lib"
else
EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
endif
endif

ifneq ($(BUILDMATHMORE),)
CXXFLAGS+= -DHAVE_MATHMORE
ifeq ($(PLATFORM),win32)
EXTRALIBS       += "$(ROOTSYS)/lib/libMathMore.lib"
else
EXTRALIBS += -lMathMore
endif
endif



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

UNURANMULTIDIMOBJ      = unuranMultiDim.$(ObjSuf)
UNURANMULTIDIMSRC      = unuranMultiDim.$(SrcSuf)
UNURANMULTIDIM         = unuranMultiDim$(ExeSuf)

UNURANDISCROBJ      = unuranDiscrete.$(ObjSuf)
UNURANDISCRSRC      = unuranDiscrete.$(SrcSuf)
UNURANDISCR         = unuranDiscrete$(ExeSuf)

UNURANHISTOBJ      = unuranHist.$(ObjSuf)
UNURANHISTSRC      = unuranHist.$(SrcSuf)
UNURANHIST         = unuranHist$(ExeSuf)




OBJS          = $(UNURANDISTROBJ)  $(UNURANSIMPLEOBJ) $(UNURANMULTI2DOBJ) $(UNURANMULTIDIMOBJ) $(UNURANDISCROBJ) $(UNURANHISTOBJ)


PROGRAMS      =  $(UNURANDISTR)  $(UNURANSIMPLE) $(UNURANMULTI2D) $(UNURANMULTIDIM) $(UNURANDISCR) $(UNURANHIST)

		  
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

$(UNURANMULTIDIM): $(UNURANMULTIDIMOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



$(UNURANSIMPLE): $(UNURANSIMPLEOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(UNURANDISCR): $(UNURANDISCROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(UNURANHIST): $(UNURANHISTOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
