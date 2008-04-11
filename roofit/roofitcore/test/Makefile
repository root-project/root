# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include Makefile.arch

#------------------------------------------------------------------------------

STRESSROOFITO    = stressRooFit.$(ObjSuf)
STRESSROOFITS    = stressRooFit.$(SrcSuf)
STRESSROOFIT     = stressRooFit$(ExeSuf)

OBJS          = $(STRESSROOFITO) 

PROGRAMS      = $(STRESSROOFIT) 

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)

all:            $(PROGRAMS)


$(STRESSROOFITO) : $(wildcard Test*.cxx )

$(STRESSROOFIT):   $(OBJS)
		   $(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		   $(MT_EXE)
		   @echo "$@ done"

clean:
		@rm -f $(OBJS) core

distclean:      clean

.SUFFIXES: .$(SrcSuf)

###

.$(SrcSuf).$(ObjSuf):
	$(CXX) -g $(CXXFLAGS) -c $<
