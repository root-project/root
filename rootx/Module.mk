# Module.mk for rootx module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := rootx
MODDIR       := $(ROOT_SRCDIR)/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTXDIR     := $(MODDIR)
ROOTXDIRS    := $(ROOTXDIR)/src
ROOTXDIRI    := $(ROOTXDIR)/inc

##### rootx #####
ROOTXH       := $(wildcard $(MODDIRI)/*.h)
ROOTXS       := $(MODDIRS)/rootx.cxx
ROOTXO       := $(call stripsrc,$(ROOTXS:.cxx=.o))
ifeq ($(BUILDCOCOA),yes)
ROOTXXS      := $(MODDIRS)/rootxx-cocoa.mm
ROOTXXO      := $(call stripsrc,$(ROOTXXS:.mm=.o))
else
ROOTXXS      := $(MODDIRS)/rootxx.cxx
ROOTXXO      := $(call stripsrc,$(ROOTXXS:.cxx=.o))
endif
ROOTXDEP     := $(ROOTXO:.o=.d) $(ROOTXXO:.o=.d)
ROOTX        := bin/root

# used in the main Makefile
ROOTXH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTXH))
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTXH_REL))
ALLEXECS     += $(ROOTX)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(ROOTXH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Roofit_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += // link no-library-created \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(ROOTXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ROOTXDIRI)/%.h
		cp $< $@

ifeq ($(BUILDCOCOA),yes)
$(ROOTX):       $(ROOTXO) $(ROOTXXO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTXO) $(ROOTXXO) $(STRLCPYO) -framework Cocoa
else
$(ROOTX):       $(ROOTXO) $(ROOTXXO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTXO) $(ROOTXXO) $(STRLCPYO) $(XLIBS)
endif

all-$(MODNAME): $(ROOTX)

clean-$(MODNAME):
		@rm -f $(ROOTXO) $(ROOTXXO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTXDEP) $(ROOTX)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifneq ($(BUILDCOCOA),yes)
$(ROOTXXO): CXXFLAGS += $(X11INCDIR:%=-I%)
else
$(ROOTXXO): OBJCXXFLAGS += $(CXXFLAGS)
endif
