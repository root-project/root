# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../..
include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(DIM),2)
CXXFLAGS += -DDIM_2
endif
ifeq ($(DIM),3)
CXXFLAGS += -DDIM_3
endif
ifneq ($(USE_CLHEP),)
CXXFLAGS += -DUSE_CLHEP
endif
ifneq ($(USE_ROOT),)
CXXFLAGS += -DUSE_ROOT
endif




# if have clhep
ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
ifeq ($(PLATFORM),win32)
EXTRALIBS += "$(CLHEPBASE)/lib/CLHEP.lib"
else
#EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
EXTRALIBS+=  -L$(CLHEPBASE)/lib -lCLHEP
endif
endif

# add MathMore lib
ifeq ($(shell $(ROOTCONFIG) --has-mathmore),yes)
ifeq ($(PLATFORM),win32)
EXTRALIBS+= '$(ROOTSYS)/lib/libMathMore.lib' 
else
EXTRALIBS+= -lMathMore
endif
endif

TESTTMATHOBJ     = testTMath.$(ObjSuf)
TESTTMATHSRC     = testTMath.$(SrcSuf)
TESTTMATH        = testTMath$(ExeSuf)

BSEARCHTIMEOBJ     = binarySearchTime.$(ObjSuf)
BSEARCHTIMESRC     = binarySearchTime.$(SrcSuf)
BSEARCHTIME        = binarySearchTime$(ExeSuf)

TESTBSEARCHOBJ     = testBinarySearch.$(ObjSuf)
TESTBSEARCHSRC     = testBinarySearch.$(SrcSuf)
TESTBSEARCH        = testBinarySearch$(ExeSuf)

TESTSORTOBJ     = stdsort.$(ObjSuf)
TESTSORTSRC     = stdsort.$(SrcSuf)
TESTSORT        = stdsort$(ExeSuf)

TESTSORTORDEROBJ     = testSortOrder.$(ObjSuf)
TESTSORTORDERSRC     = testSortOrder.$(SrcSuf)
TESTSORTORDER        = testSortOrder$(ExeSuf)

SPECFUNERFOBJ     = testSpecFuncErf.$(ObjSuf)
SPECFUNERFSRC     = testSpecFuncErf.$(SrcSuf)
SPECFUNERF        = testSpecFuncErf$(ExeSuf)

SPECFUNGAMMAOBJ     = testSpecFuncGamma.$(ObjSuf)
SPECFUNGAMMASRC     = testSpecFuncGamma.$(SrcSuf)
SPECFUNGAMMA        = testSpecFuncGamma$(ExeSuf)

SPECFUNBETAOBJ     = testSpecFuncBeta.$(ObjSuf)
SPECFUNBETASRC     = testSpecFuncBeta.$(SrcSuf)
SPECFUNBETA        = testSpecFuncBeta$(ExeSuf)

SPECFUNBETAIOBJ     = testSpecFuncBetaI.$(ObjSuf)
SPECFUNBETAISRC     = testSpecFuncBetaI.$(SrcSuf)
SPECFUNBETAI        = testSpecFuncBetaI$(ExeSuf)

SPECFUNSICIOBJ     = testSpecFuncSiCi.$(ObjSuf)
SPECFUNSICISRC     = testSpecFuncSiCi.$(SrcSuf)
SPECFUNSICI        = testSpecFuncSiCi$(ExeSuf)

STRESSTMATHOBJ     = stressTMath.$(ObjSuf)
STRESSTMATHSRC     = stressTMath.$(SrcSuf)
STRESSTMATH        = stressTMath$(ExeSuf)


STRESSTF1OBJ     = stressTF1.$(ObjSuf)
STRESSTF1SRC     = stressTF1.$(SrcSuf)
STRESSTF1        = stressTF1$(ExeSuf)

STRESSGOFOBJ     = stressGoFTest.$(ObjSuf)
STRESSGOFSRC     = stressGoFTest.$(SrcSuf)
STRESSGOF        = stressGoFTest$(ExeSuf)


INTEGRATIONOBJ      = testIntegration.$(ObjSuf)
INTEGRATIONSRC      = testIntegration.$(SrcSuf)
INTEGRATION         = testIntegration$(ExeSuf)

INTEGRATIONMULTIOBJ = testIntegrationMultiDim.$(ObjSuf)
INTEGRATIONMULTISRC = testIntegrationMultiDim.$(SrcSuf)
INTEGRATIONMULTI    = testIntegrationMultiDim$(ExeSuf)

ROOTFINDEROBJ      = testRootFinder.$(ObjSuf)
ROOTFINDERSRC      = testRootFinder.$(SrcSuf)
ROOTFINDER         = testRootFinder$(ExeSuf)

DISTSAMPLEROBJ      = testDistSampler.$(ObjSuf)
DISTSAMPLERSRC      = testDistSampler.$(SrcSuf)
DISTSAMPLER         = testDistSampler$(ExeSuf)

KDTREEOBJ          = kDTreeTest.$(ObjSuf)
KDTREESRC          = kDTreeTest.$(SrcSuf)
KDTREE             = kDTreeTest

NEWKDTREEOBJ          = newKDTreeTest.$(ObjSuf)
NEWKDTREESRC          = newKDTreeTest.$(SrcSuf)
NEWKDTREE             = newKDTreeTest

OBJS          = $(SPECFUNBETAOBJ) $(SPECFUNBETAIOBJ) $(SPECFUNGAMMAOBJ) $(SPECFUNCISIOBJ) $(SPECFUNERFOBJ) $(TESTTMATHOBJ) $(BSEARCHTIMEOBJ)  $(TESTBSEARCHOBJ)  $(TESTSORTOBJ) $(TESTSORTORDEROBJ) $(STRESSTMATHOBJ) $(STRESSTF1OBJ) $(INTEGRATIONOBJ) $(INTEGRATIONMULTIOBJ) $(ROOTFINDEROBJ) $(DISTSAMPLEROBJ) $(KDTREEOBJ) $(NEWKDTREEOBJ)


PROGRAMS      =$(SPECFUNBETA) $(SPECFUNBETAI)  $(SPECFUNGAMMA) $(SPECFUNSICI) $(SPECFUNERF) $(TESTTMATH) $(BSEARCHTIME) $(TESTBSEARCH) $(TESTSORT) $(TESTSORTORDER) $(STRESSTMATH) $(STRESSTF1) $(ITERATOR)  $(INTEGRATION) $(INTEGRATIONMULTI) $(ROOTFINDER) $(DISTSAMPLER) $(KDTREE) $(NEWKDTREE)


.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS) 

$(STRESSTF1):      $(STRESSTF1OBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"


$(STRESSTMATH):     $(STRESSTMATHOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTTMATH):      $(TESTTMATHOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(BSEARCHTIME):      $(BSEARCHTIMEOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTBSEARCH):      $(TESTBSEARCHOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTSORT):      $(TESTSORTOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTSORTORDER):      $(TESTSORTORDEROBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(SPECFUNERF):      $(SPECFUNERFOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(SPECFUNGAMMA):    $(SPECFUNGAMMAOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(SPECFUNSICI):    $(SPECFUNSICIOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(SPECFUNBETA):    $(SPECFUNBETAOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(SPECFUNBETAI):    $(SPECFUNBETAIOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(INTEGRATION): $(INTEGRATIONOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		@echo "$@ done"

$(INTEGRATIONMULTI): $(INTEGRATIONMULTIOBJ)
			$(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
			@echo "$@ done"

$(ROOTFINDER): $(ROOTFINDEROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(DISTSAMPLER): $(DISTSAMPLEROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		@echo "$@ done"

$(KDTREE):	$(KDTREEOBJ)
		 $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(NEWKDTREE):	$(NEWKDTREEOBJ)
		 $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESSGOF):      $(STRESSGOFOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"




clean:
		@rm -f $(OBJS) core TrackDict.$(SrcSuf) TrackDict.h

distclean:      clean
		@rm -f $(PROGRAMS) libTrackDict.*


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
