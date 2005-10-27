# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
EXTRALIBS        = $(ROOTSYS)/lib/libGMinuit.lib $(ROOTSYS)/lib/libMathCore.lib
else
EXTRALIBS        = -lGMinuit 
endif

USERFUNCOBJ      = testUserFunc.$(ObjSuf)
USERFUNCSRC      = testUserFunc.$(SrcSuf)
USERFUNC         = testUserFunc$(ExeSuf)


# ANDYOBJ      = testAndy.$(ObjSuf)
# ANDYSRC      = testAndy.$(SrcSuf)
# ANDY         = testAndy$(ExeSuf)

GRAPHOBJ      = testGraph.$(ObjSuf)
GRAPHSRC      = testGraph.$(SrcSuf)
GRAPH        = testGraph$(ExeSuf)


OBJS          = $(USERFUNCOBJ)  $(GRAPHOBJ)
#                 $(TSTRINGO) $(TCOLLEXO) $(VVECTORO) $(VMATRIXO) $(VLAZYO) \
#                 $(HELLOO) $(ACLOCKO) $(STRESSO) $(TBENCHO) $(BENCHO) \
#                 $(TCOLLBMO) $(STRESSGEOMO) $(STRESSLO)  $(TESTBITSO) \
#                 $(CTORTUREO) $(QPRANDOMO) $(THREADO) $(STRESSVECO)

PROGRAMS      = $(USERFUNC)  $(GRAPH)
#                 $(TCOLLEX) $(TCOLLBM) $(VVECTOR) $(VMATRIX) $(VLAZY) \
#                 $(HELLOSO) $(ACLOCKSO) $(STRESS) $(TBENCHSO) $(BENCH) \
#                 $(STRESSGEOM) $(STRESSL) $(TESTBITS) $(CTORTURE) $(QPRANDOM) \
#                 $(THREADS) $(STRESSVEC)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)

$(USERFUNC):    $(USERFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

# $(ANDY):        $(ANDYOBJ)
# 		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
# 		@echo "$@ done"

$(GRAPH): 	$(GRAPHOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) $(EVENTSO) $(EVENTLIB) *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations
		@rm -rf cxx_repository
		-@cd RootShower && $(MAKE) distclea

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
