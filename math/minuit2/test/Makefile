# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../..
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
EXTRALIBS        = "$(ROOTSYS)/lib/libMinuit2.lib"
else
EXTRALIBS        = -lMinuit2 
CXXFLAGS += -g      
endif

# for using with MPI
ifneq ($(USE_MPI),)
CXX=mpic++
LD=mpic++
endif
ifneq ($(USE_OPENMP),)
CXXFLAGS += -D_GLIBCXX_PARALLEL -fopenmp 
LDFLAGS += -fopenmp 
endif



USERFUNCOBJ      = testUserFunc.$(ObjSuf)
USERFUNCSRC      = testUserFunc.$(SrcSuf)
USERFUNC         = testUserFunc$(ExeSuf)

MINIMIZEROBJ      = testMinimizer.$(ObjSuf)
MINIMIZERSRC      = testMinimizer.$(SrcSuf)
MINIMIZER         = testMinimizer$(ExeSuf)

NDIMFITOBJ      = testNdimFit.$(ObjSuf)
NDIMFITSRC      = testNdimFit.$(SrcSuf)
NDIMFIT         = testNdimFit$(ExeSuf)

GAUSFITOBJ      = testUnbinGausFit.$(ObjSuf)
GAUSFITSRC      = testUnbinGausFit.$(SrcSuf)
GAUSFIT         = testUnbinGausFit$(ExeSuf)


OBJS          = $(USERFUNCOBJ)  $(GRAPHOBJ)  $(MINIMIZEROBJ) $(NDIMFITOBJ) $(GAUSFITOBJ)

PROGRAMS      = $(USERFUNC)  $(GRAPH)  $(MINIMIZER) $(NDIMFIT) $(GAUSFIT)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)

$(USERFUNC):    $(USERFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(MINIMIZER): 	$(MINIMIZEROBJ)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(LIBS)  "$(ROOTSYS)/lib/libMinuit2.lib" $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(LIBS) -lMathCore  $(OutPutOpt)$@
endif
		@echo "$@ done"

$(NDIMFIT): 	$(NDIMFITOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(GAUSFIT): 	$(GAUSFITOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"


clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS)  *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
