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
#unroll loop better on gcc > 4
CXXFLAGS+= -O3 -g  -funroll-loops 
endif

# if have clhep
#CLHEPBASE=/Users/moneta/mathlibs/CLHEP-1.9.2.2
ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
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

testOperations.$(ObjSuf): matrix_op.h matrix_util.h TestTimer.h

testKalman.$(ObjSuf): matrix_util.h TestTimer.h

stressOperations.$(ObjSuf): $(TESTOPERATIONSOBJ)


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
		@rm -f $(PROGRAMS) kalman.root stressOperations.root 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):  
	$(CXX)  $(CXXFLAGS) -c $<
