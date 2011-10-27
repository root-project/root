# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

RC := root-config
ifeq ($(shell which $(RC) 2>&1 | sed -ne "s@.*/$(RC)@$(RC)@p"),$(RC))
include $(shell $(RC) --etcdir)/Makefile.arch
else
ROOTSYS = ../..
include $(ROOTSYS)/etc/Makefile.arch
endif

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
EXTRALIBS   = "$(ROOTSYS)/lib/libGuiHtml.lib"
LDOPT      += /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
ifneq (,$(findstring $(VC_MAJOR),14 15))
MT_MANIFEST = mt -nologo -manifest $(PROGRAMS).manifest -outputresource:$(PROGRAMS)\;2
else
MT_MANIFEST = 
endif
else
EXTRALIBS   = -lGuiHtml
MT_MANIFEST = 
endif

RHTMLS        := rhtml.$(SrcSuf) rhtmlDict.$(SrcSuf)
RHTMLO        := $(RHTMLS:.$(SrcSuf)=.$(ObjSuf))
RHTML          = rhtml$(ExeSuf)

OBJS          = $(RHTMLO)

PROGRAMS      = $(RHTML)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)

all:            $(PROGRAMS)

$(RHTML):     $(RHTMLO)
	$(LD) $(LDFLAGS) $^ $(GLIBS) $(EXTRALIBS) $(OutPutOpt)$@
	$(MT_MANIFEST)
	@echo "$@ done"

#########################################

rhtmlDict.$(SrcSuf): rhtml.h LinkDef.h
		@echo "Generating dictionary $@..."
		@rootcint -f $@ -c $^

clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS)  *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.pdb *.ilk \
		   *.manifest *.d *.def

.SUFFIXES: .$(SrcSuf)

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
