# Makefile for the RootShower program.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/2/2003

include ../Makefile.arch

#------------------------------------------------------------------------------

ROOTSHOWERS  := $(wildcard *.$(SrcSuf)) RootShowerDict.$(SrcSuf)
ROOTSHOWERO  := $(ROOTSHOWERS:.$(SrcSuf)=.$(ObjSuf))
ROOTSHOWER    = RootShower$(ExeSuf)

OBJS          = $(ROOTSHOWERO)

PROGRAMS      = $(ROOTSHOWER)

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)

all:            $(PROGRAMS)

$(ROOTSHOWER):  $(ROOTSHOWERO)
ifeq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) $^ $(GLIBS) '$(ROOTSYS)\lib\libEG.lib' \
		   '$(ROOTSYS)\lib\libHtml.lib' '$(ROOTSYS)\lib\libGeom.lib' $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(GLIBS) -lEG -lHtml -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

RootShowerDict.$(SrcSuf): MyParticle.h MyDetector.h MyEvent.h RSLinkDef.h
		@echo "Generating dictionary $@..."
		@rootcint -f $@ -c $^

clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) *Dict.* *.def *.exp \
		   *.geom *.root *.ps *.so .def so_locations
		@rm -rf cxx_repository

###

.$(SrcSuf).$(ObjSuf):
		$(CXX) $(CXXFLAGS) -c $<
