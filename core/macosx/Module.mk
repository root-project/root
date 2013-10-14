# Module.mk for macosx module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Timur Pocheptsov, 5/12/2011

MACOSXNDEBUG := -DNDEBUG
ifeq ($(ROOTBUILD),debug)
   MACOSXNDEBUG :=
endif


MODNAME      := macosx
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MACOSXDIR    := $(MODDIR)
MACOSXDIRS   := $(MACOSXDIR)/src
MACOSXDIRI   := $(MACOSXDIR)/inc

##### libMacOSX  (part of libCore) #####
MACOSXL	     := $(MODDIRI)/LinkDef.h

MACOSXH1     := $(wildcard $(MODDIRI)/T*.h)
MACOSXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MACOSXS      := $(wildcard $(MODDIRS)/*.mm)
MACOSXO      := $(call stripsrc,$(MACOSXS:.mm=.o))

MACOSXDEP    := $(MACOSXO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MACOSXH))

# include all dependency files
INCLUDEFILES += $(MACOSXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MACOSXDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(MACOSXO)

clean-$(MODNAME):
		@rm -f $(MACOSXO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MACOSXDEP)

distclean::     distclean-$(MODNAME)

$(MACOSXO): CXXFLAGS += $(MACOSXNDEBUG)
