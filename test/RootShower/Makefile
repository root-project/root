# Makefile for the RootShower program.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/2/2003

RC     := root-config
ifeq ($(shell which $(RC) 2>&1 | sed -ne "s@.*/$(RC)@$(RC)@p"),$(RC))
MKARCH := $(wildcard $(shell $(RC) --etcdir)/Makefile.arch)
endif
ifneq ($(MKARCH),)
include $(MKARCH)
else
ROOTSYS = ../..
include $(ROOTSYS)/etc/Makefile.arch
endif

#------------------------------------------------------------------------------

ROOTSHOWERS  := $(wildcard *.$(SrcSuf)) RootShowerDict.$(SrcSuf)
ROOTSHOWERO  := $(ROOTSHOWERS:.$(SrcSuf)=.$(ObjSuf))
ROOTSHOWER    = RootShower$(ExeSuf)
ifeq ($(PLATFORM),win32)
ROOTSHOWERRC  := rootshower.rc
ROOTSHOWERRES := rootshower.res
endif

OBJS          = $(ROOTSHOWERO)

PROGRAMS      = $(ROOTSHOWER)

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)

all:            $(PROGRAMS)

ifeq ($(PLATFORM),win32)
$(ROOTSHOWERRES) : $(ROOTSHOWERRC)
	rc -DBUILDNUMBER=0 -r -fo $(ROOTSHOWERRES) $(ROOTSHOWERRC)
endif


ifeq ($(PLATFORM),win32)
$(ROOTSHOWER):  $(ROOTSHOWERO) $(ROOTSHOWERRES)
		$(LD) $(LDFLAGS) /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup \
		$^ $(GLIBS) '$(ROOTSYS)\lib\libEG.lib' '$(ROOTSYS)\lib\libHtml.lib' \
		'$(ROOTSYS)\lib\libGeom.lib' $(OutPutOpt)$@
		$(MT_EXE)
else
$(ROOTSHOWER):  $(ROOTSHOWERO)
		$(LD) $(LDFLAGS) $^ $(GLIBS) -lEG -lGed -lHtml -lThread -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

RootShowerDict.$(SrcSuf): MyParticle.h MyDetector.h MyEvent.h RSLinkDef.h
		@echo "Generating dictionary $@..."
		@rootcint -f $@ -c $^

clean:
		@rm -f $(OBJS) *.res core

distclean:      clean
		@rm -f $(PROGRAMS) *Dict.* *.def *.exp \
		   *.geom *.root *.ps *.so .def so_locations
		@rm -rf cxx_repository

###

.$(SrcSuf).$(ObjSuf):
		$(CXX) $(CXXFLAGS) -c $<
