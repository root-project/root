# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include ../../test/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),macosx) 
CXXFLAGS+= -g
endif

ifneq ($(PLATFORM),win32) 
LIBM = -lm
endif



TESTSMATRIXOBJ     = testSMatrix.$(ObjSuf)
TESTSMATRIXSRC     = testSMatrix.$(SrcSuf)
TESTSMATRIX        = testSMatrix$(ExeSuf)

TESTOPERATIONSOBJ     = testOperations.$(ObjSuf)
TESTOPERATIONSSRC     = testOperations.$(SrcSuf)
TESTOPERATIONS        = testOperations$(ExeSuf)

TESTKALMANOBJ     = testKalman.$(ObjSuf)
TESTKALMANSRC     = testKalman.$(SrcSuf)
TESTKALMAN        = testKalman$(ExeSuf)

TESTIOOBJ     = testIO.$(ObjSuf)
TESTIOSRC     = testIO.$(SrcSuf)
TESTIO        = testIO$(ExeSuf)


STRESSOPERATIONSOBJ     = stressOperations.$(ObjSuf)
STRESSOPERATIONSSRC     = stressOperations.$(SrcSuf)
STRESSOPERATIONS        = stressOperations$(ExeSuf)


STRESSKALMANOBJ     = stressKalman.$(ObjSuf)
STRESSKALMANSRC     = stressKalman.$(SrcSuf)
STRESSKALMAN        = stressKalman$(ExeSuf)


OBJS          = $(TESTSMATRIXOBJ) $(TESTOPERATIONSOBJ) $(TESTKALMANOBJ) $(TESTIOOBJ) $(STRESSOPERATIONSOBJ) $(STRESSKALMANOBJ)


PROGRAMS      = $(TESTSMATRIX)  $(TESTOPERATIONS) $(TESTKALMAN) $(TESTIO) $(STRESSOPERATIONS) $(STRESSKALMAN) 


.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)


$(TESTSMATRIX):   $(TESTSMATRIXOBJ)
		    $(LD) $(LDFLAGS) $^  $(LIBM) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTOPERATIONS):   $(TESTOPERATIONSOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTKALMAN):    $(TESTKALMANOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTIO):        $(TESTIOOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESSKALMAN):  $(STRESSKALMANOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESSOPERATIONS):  $(STRESSOPERATIONSOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"



clean:
		@rm -f $(OBJS) $(PROGRAMS) core 

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX)  $(CXXFLAGS) -c $<
