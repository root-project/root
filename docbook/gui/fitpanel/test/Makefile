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

CXXFLAGS+=  -I$(ROOTSYS)/gui/fitpanel/src

LDFLAGS+=-lFitPanel -lGui

UNITTESTOBJ     = UnitTesting.$(ObjSuf)
UNITTESTSRC     = UnitTesting.$(SrcSuf)
UNITTEST        = UnitTesting$(ExeSuf)


OBJS          = $(UNITTESTOBJ)

PROGRAMS      =$(UNITTEST)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS) 

$(UNITTEST):      $(UNITTESTOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

clean:
		@rm -f $(OBJS) outputUnitTesting.txt

distclean:      clean
		@rm -f $(PROGRAMS)


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
