# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../../..
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
LIBS        += "$(ROOTSYS)/lib/libMinuit2.lib"
else
LIBS        += -lMinuit2 
CXXFLAGS += -g      
endif

# for using with MPI
ifneq ($(USE_OPENMPI),)
CXX=mpic++
LD=mpic++
endif

DEMOGAUSSSIMSRC    = DemoGaussSim.$(SrcSuf) GaussDataGen.$(SrcSuf) GaussFcn.$(SrcSuf)
DEMOGAUSSSIMOBJ    = DemoGaussSim.$(ObjSuf) GaussDataGen.$(ObjSuf) GaussFcn.$(ObjSuf)
DEMOGAUSSSIM       = test_Minuit2_DemoGaussSim$(ExeSuf)

DEMOFUMILISRC     = DemoFumili.$(SrcSuf) GaussDataGen.$(SrcSuf) 
DEMOFUMILIOBJ     = DemoFumili.$(ObjSuf) GaussDataGen.$(ObjSuf) 
DEMOFUMILI        = test_Minuit2_DemoFumili$(ExeSuf)

PTESTSRC    = PaulTest.$(SrcSuf) GaussFcn.$(SrcSuf)
PTESTOBJ    = PaulTest.$(ObjSuf) GaussFcn.$(ObjSuf)
PTEST       = test_Minuit2_PaulTest$(ExeSuf)

PTEST2SRC    = PaulTest2.$(SrcSuf) GaussFcn.$(SrcSuf)
PTEST2OBJ    = PaulTest2.$(ObjSuf) GaussFcn.$(ObjSuf)
PTEST2       = test_Minuit2_PaulTest2$(ExeSuf)

PTEST3SRC    = PaulTest3.$(SrcSuf) GaussFcn2.$(SrcSuf)
PTEST3OBJ    = PaulTest3.$(ObjSuf) GaussFcn2.$(ObjSuf)
PTEST3       = test_Minuit2_PaulTest3$(ExeSuf)

PTEST4SRC    = PaulTest4.$(SrcSuf)
PTEST4OBJ    = PaulTest4.$(ObjSuf)
PTEST4       = test_Minuit2_PaulTest4$(ExeSuf)

RTESTSRC    = ReneTest.$(SrcSuf)
RTESTOBJ    = ReneTest.$(ObjSuf)
RTEST       = test_Minuit2_ReneTest$(ExeSuf)

PARATESTSRC    = ParallelTest.$(SrcSuf) GaussDataGen.$(SrcSuf)
PARATESTOBJ    = ParallelTest.$(ObjSuf) GaussDataGen.$(ObjSuf)
PARATEST       = test_Minuit2_Parallel$(ExeSuf)


OBJS          = $(DEMOGAUSSSIMOBJ) $(DEMOFUMILIOBJ) $(PTESTOBJ) $(PTEST2OBJ) $(PTEST3OBJ) $(PTEST4OBJ) $(RTESTOBJ) $(PARATESTOBJ)

PROGRAMS      = $(DEMOGAUSSSIM) $(DEMOFUMILI) $(PTEST) $(PTEST2) $(PTEST3) $(PTEST4) $(RTEST) $(PARATEST)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)

$(DEMOGAUSSSIMOBJ) : GaussFunction.h GaussRandomGen.h GaussDataGen.h FlatRandomGen.h GaussFcn.h
$(DEMOGAUSSSIM): 	$(DEMOGAUSSSIMOBJ)
			$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
			@echo "$@ done"

$(DEMOFUMILIOBJ) : GaussianModelFunction.h GaussRandomGen.h GaussDataGen.h FlatRandomGen.h 
$(DEMOFUMILI): 	$(DEMOFUMILIOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(PTESTOBJ): 	GaussFcn.h
$(PTEST): 	$(PTESTOBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(PTEST2OBJ): 	GaussFcn.h
$(PTEST2): 	$(PTEST2OBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(PTEST3OBJ) : 	GaussFcn2.h 
$(PTEST3): 	$(PTEST3OBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(PTEST4): 	$(PTEST4OBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(RTEST): 	$(RTESTOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(PARATESTOBJ): 	GaussRandomGen.h
$(PARATEST): 	$(PARATESTOBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"


clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS)  *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
