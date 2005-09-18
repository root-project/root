# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

#EXTRALIBS        = -llcg_Minuit -lGMinuit -lMathMore
EXTRALIBS        = -lMathMore -lMathCore

USERFUNCOBJ      = testChebyshev.$(ObjSuf)
USERFUNCSRC      = testChebyshev.$(SrcSuf)
USERFUNC         = testChebyshev$(ExeSuf)


OBJS          = $(USERFUNCOBJ) 
#                 $(TSTRINGO) $(TCOLLEXO) $(VVECTORO) $(VMATRIXO) $(VLAZYO) \
#                 $(HELLOO) $(ACLOCKO) $(STRESSO) $(TBENCHO) $(BENCHO) \
#                 $(TCOLLBMO) $(STRESSGEOMO) $(STRESSLO)  $(TESTBITSO) \
#                 $(CTORTUREO) $(QPRANDOMO) $(THREADO) $(STRESSVECO)

PROGRAMS      = $(USERFUNC)
#                 $(TCOLLEX) $(TCOLLBM) $(VVECTOR) $(VMATRIX) $(VLAZY) \
#                 $(HELLOSO) $(ACLOCKSO) $(STRESS) $(TBENCHSO) $(BENCH) \
#                 $(STRESSGEOM) $(STRESSL) $(TESTBITS) $(CTORTURE) $(QPRANDOM) \
#                 $(THREADS) $(STRESSVEC)




$(USERFUNC):    $(USERFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) $(EVENTSO) $(EVENTLIB) *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations
		@rm -rf cxx_repository
		-@cd RootShower && $(MAKE) distclean

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
