# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

EXTRALIBS        = -lMathMore -lMathCore

CHEBYSHEVOBJ      = testChebyshev.$(ObjSuf)
CHEBYSHEVSRC      = testChebyshev.$(SrcSuf)
CHEBYSHEV         = testChebyshev$(ExeSuf)

DERIVATIONOBJ      = testDerivation.$(ObjSuf)
DERIVATIONSRC      = testDerivation.$(SrcSuf)
DERIVATION         = testDerivation$(ExeSuf)

INTEGRATIONOBJ      = testIntegration.$(ObjSuf)
INTEGRATIONSRC      = testIntegration.$(SrcSuf)
INTEGRATION         = testIntegration$(ExeSuf)

ROOTFINDEROBJ      = testRootFinder.$(ObjSuf)
ROOTFINDERSRC      = testRootFinder.$(SrcSuf)
ROOTFINDER         = testRootFinder$(ExeSuf)

INTERPOLATIONOBJ      = testInterpolation.$(ObjSuf)
INTERPOLATIONSRC      = testInterpolation.$(SrcSuf)
INTERPOLATION         = testInterpolation$(ExeSuf)



OBJS          = $(CHEBYSHEVOBJ) $(DERIVATIONOBJ) $(INTEGRATIONOBJ) $(ROOTFINDEROBJ) \
                  $(INTERPOLATIONOBJ)

PROGRAMS      = $(CHEBYSHEV)  $(DERIVATION) $(INTEGRATION) $(ROOTFINDER) \
                  $(INTERPOLATION)


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

$(ROOTFINDER):  $(ROOTFINDEROBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(INTERPOLATION):  $(INTERPOLATIONOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"


clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
