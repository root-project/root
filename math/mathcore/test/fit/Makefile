# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../../..
include $(ROOTSYS)/config/Makefile.config
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------



ifeq ($(PLATFORM),win32)
EXTRALIBS= "$(ROOTSYS)/lib/libMathMore.lib" 
ROOFITLIBS= "$(ROOTSYS)/lib/libMinuit.lib" "$(ROOTSYS)/lib/libThread.lib" "$(ROOTSYS)/lib/libRooFit.lib" "$(ROOTSYS)/lib/libRooFitCore.lib" "$(ROOTSYS)/lib/libHtml.lib"
else
EXTRALIBS=  
ROOFITLIBS= -lMathMore -lMinuit  -lThread -lRooFit -lRooFitCore -lHtml -lFoam
CXXFLAGS += -g
endif

SPARSEDATACOMPAREROBJ      = SparseDataComparer.$(ObjSuf)
SPARSEDATACOMPARERSRC      = SparseDataComparer.$(SrcSuf)
SPARSEDATACOMPARER         = SparseDataComparer$(ExeSuf)

SPARSEFIT3OBJ      = SparseFit3.$(ObjSuf)
SPARSEFIT3SRC      = SparseFit3.$(SrcSuf)
SPARSEFIT3         = SparseFit3$(ExeSuf)

SPARSEFIT4OBJ      = SparseFit4.$(ObjSuf)
SPARSEFIT4SRC      = SparseFit4.$(SrcSuf)
SPARSEFIT4         = SparseFit4$(ExeSuf)

TESTFITOBJ      = testFit.$(ObjSuf)
TESTFITSRC      = testFit.$(SrcSuf)
TESTFIT         = testFit$(ExeSuf)


TESTMINIMOBJ      = testMinim.$(ObjSuf)
TESTMINIMSRC      = testMinim.$(SrcSuf)
TESTMINIM         = testMinim$(ExeSuf)

ifeq ($(shell $(ROOTCONFIG) --has-mathmore),yes)
ifeq ($(shell $(ROOTCONFIG) --has-roofit),yes)

TESTPERFOBJ      = testFitPerf.$(ObjSuf)
TESTPERFSRC      = testFitPerf.$(SrcSuf)
TESTPERF         = testFitPerf$(ExeSuf)

TESTROOFITOBJ      = testRooFit.$(ObjSuf)
TESTROOFITSRC      = testRooFit.$(SrcSuf)
TESTROOFIT         = testRooFit$(ExeSuf)

endif
endif




OBJS          = $(SPARSEFIT4OBJ) $(SPARSEFIT3OBJ) $(SPARSEDATACOMPAREROBJ) $(TESTFITOBJ) $(TESTPERFOBJ) $(TESTMINIMOBJ) $(TESTROOFITOBJ)


PROGRAMS      =  $(SPARSEFIT4) $(SPARSEFIT3) $(SPARSEDATACOMPARER) $(TESTFIT) $(TESTPERF) $(TESTMINIM) $(TESTROOFIT)

		  
.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)

all:            $(PROGRAMS)

$(SPARSEFIT4): $(SPARSEFIT4OBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(SPARSEFIT3): $(SPARSEFIT3OBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(SPARSEDATACOMPARER): $(SPARSEDATACOMPAREROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTFIT): $(TESTFITOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTPERFOBJ): GaussFunction.h
$(TESTPERF): $(TESTPERFOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(ROOFITLIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTMINIM): $(TESTMINIMOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTROOFITOBJ): WrapperRooPdf.h
$(TESTROOFIT): $(TESTROOFITOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(ROOFITLIBS) $(OutPutOpt)$@
		@echo "$@ done"


clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
