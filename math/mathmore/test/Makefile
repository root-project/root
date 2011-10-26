# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000


include $(ROOTSYS)/config/Makefile.config
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------


CXXFLAGS += $(GSLFLAGS)


ifeq ($(PLATFORM),win32)
EXTRALIBS        = "$(ROOTSYS)/lib/libMathMore.lib" "$(ROOTSYS)/lib/libMathCore.lib"  $(GSLLIBS)
UNURANLIB        = "$(ROOTSYS)/lib/libUnuran.lib"
TMVALIB          = "$(ROOTSYS)/lib/libTMVA.lib" "$(ROOTSYS)/lib/libMLP.lib" "$(ROOTSYS)/lib/libTreePlayer.lib" \
	           "$(ROOTSYS)/lib/libMinuit.lib" "$(ROOTSYS)/lib/libMinuit2.lib"

else
EXTRALIBS        = -lMathMore -lMathCore $(GSLLIBDIR) $(GSLLIBS)
UNURANLIB        = -lUnuran
TMVALIB          = -lTMVA -lMLP -lTreePlayer -lMinuit -lMinuit2
CXXFLAGS += -g

ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
endif

endif


CXXFLAGS += -DHAVE_ROOTLIBS

#--- gsl libraries
#CXXFLAGS += `gsl-config --cflags`
#EXTRALIBS += `gsl-config --libs`
#---

CHEBYSHEVOBJ      = testChebyshev.$(ObjSuf)
CHEBYSHEVSRC      = testChebyshev.$(SrcSuf)
CHEBYSHEV         = testChebyshev$(ExeSuf)

TESTPERMUTEOBJ     = testPermute.$(ObjSuf)
TESTPERMUTESRC     = testPermute.$(SrcSuf)
TESTPERMUTE        = testPermute$(ExeSuf)

DERIVATIONOBJ      = testDerivation.$(ObjSuf)
DERIVATIONSRC      = testDerivation.$(SrcSuf)
DERIVATION         = testDerivation$(ExeSuf)

INTEGRATIONOBJ      = testIntegration.$(ObjSuf)
INTEGRATIONSRC      = testIntegration.$(SrcSuf)
INTEGRATION         = testIntegration$(ExeSuf)

INTEGRATIONMDOBJ      = testIntegrationMultiDim.$(ObjSuf)
INTEGRATIONMDSRC      = testIntegrationMultiDim.$(SrcSuf)
INTEGRATIONMD         = testIntegrationMultiDim$(ExeSuf)

ROOTFINDEROBJ      = testRootFinder.$(ObjSuf)
ROOTFINDERSRC      = testRootFinder.$(SrcSuf)
ROOTFINDER         = testRootFinder$(ExeSuf)

MULTIROOTFINDEROBJ      = testMultiRootFinder.$(ObjSuf)
MULTIROOTFINDERSRC      = testMultiRootFinder.$(SrcSuf)
MULTIROOTFINDER         = testMultiRootFinder$(ExeSuf)

MINIMIZATION1DOBJ      = testMinimization1D.$(ObjSuf)
MINIMIZATION1DSRC      = testMinimization1D.$(SrcSuf)
MINIMIZATION1D         = testMinimization1D$(ExeSuf)

INTERPOLATIONOBJ      = testInterpolation.$(ObjSuf)
INTERPOLATIONSRC      = testInterpolation.$(SrcSuf)
INTERPOLATION         = testInterpolation$(ExeSuf)

RANDOMOBJ      = testRandom.$(ObjSuf)
RANDOMSRC      = testRandom.$(SrcSuf)
RANDOM         = testRandom$(ExeSuf)

RANDOMDISTOBJ      = testRandomDist.$(ObjSuf)
RANDOMDISTSRC      = testRandomDist.$(SrcSuf)
RANDOMDIST         = testRandomDist$(ExeSuf)

SPECFUNCOBJ      = testSpecFunc.$(ObjSuf)
SPECFUNCSRC      = testSpecFunc.$(SrcSuf)
SPECFUNC         = testSpecFunc$(ExeSuf)

STATFUNCOBJ      = testStatFunc.$(ObjSuf)
STATFUNCSRC      = testStatFunc.$(SrcSuf)
STATFUNC         = testStatFunc$(ExeSuf)

TESTFUNCTOROBJ      = testFunctor.$(ObjSuf)
TESTFUNCTORSRC      = testFunctor.$(SrcSuf)
TESTFUNCTOR         = testFunctor$(ExeSuf)

TESTVAVILOVOBJ      = testVavilov.$(ObjSuf)
TESTVAVILOVSRC      = testVavilov.$(SrcSuf)
TESTVAVILOV         = testVavilov$(ExeSuf)


SIMANTSPOBJ      = simanTSP.$(ObjSuf)
SIMANTSPSRC      = simanTSP.$(SrcSuf)
SIMANTSP         = simanTSP$(ExeSuf)



OBJS          = $(CHEBYSHEVOBJ) $(DERIVATIONOBJ) $(INTEGRATIONOBJ) $(INTEGRATIONMDOBJ) $(ROOTFINDEROBJ) $(MULTIROOTFINDEROBJ)\
                  $(MINIMIZATION1DOBJ) $(INTERPOLATIONOBJ)  $(RANDOMOBJ) $(RANDOMDISTOBJ) \
	          $(SPECFUNCOBJ) $(STATFUNCOBJ) $(TESTFUNCTOROBJ)  $(TESTPERMUTEOBJ) $(TESTVAVILOVOBJ) $(SIMANTSPOBJ)

PROGRAMS      = $(CHEBYSHEV)  $(DERIVATION) $(INTEGRATION) $(INTEGRATIONMD) $(ROOTFINDER) $(MULTIROOTFINDER) \
                  $(MINIMIZATION1D) $(INTERPOLATION) $(RANDOM) $(RANDOMDIST)  \
	          $(SPECFUNC) $(STATFUNC) $(TESTFUNCTOR) $(TESTPERMUTE) $(TESTVAVILOV) $(SIMANTSP)
		  
.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)


$(CHEBYSHEV):   $(CHEBYSHEVOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(DERIVATION):  $(DERIVATIONOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(INTEGRATION): $(INTEGRATIONOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(INTEGRATIONMD): $(INTEGRATIONMDOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(ROOTFINDER):  $(ROOTFINDEROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(MULTIROOTFINDER):  $(MULTIROOTFINDEROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(MINIMIZATION1D):  $(MINIMIZATION1DOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(INTERPOLATION):  $(INTERPOLATIONOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(RANDOM):  	$(RANDOMOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

testRandomDist.$(ObjSuf):       UnuRanDist.h


$(RANDOMDIST):  $(RANDOMDISTOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(UNURANLIB) $(OutPutOpt)$@
		@echo "$@ done"

$(SPECFUNC):  	$(SPECFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"


$(STATFUNC):  	$(STATFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"


$(TESTFUNCTOR): $(TESTFUNCTOROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(TESTPERMUTE):      $(TESTPERMUTEOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"


$(TESTVAVILOVOBJ):       VavilovTest.cxx VavilovTest.h

$(TESTVAVILOV): $(TESTVAVILOVOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"


$(SIMANTSP):  	$(SIMANTSPOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
