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
ROOTXS       := $(wildcard $(MODDIRS)/*.cxx)
ROOTXO       := $(call stripsrc,$(ROOTXS:.cxx=.o))
ROOTXDEP     := $(ROOTXO:.o=.d)
ROOTX        := bin/root

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTXH))
ALLEXECS     += $(ROOTX)

# include all dependency files
INCLUDEFILES += $(ROOTXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ROOTXDIRI)/%.h
		cp $< $@

$(ROOTX):       $(ROOTXO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTXO) $(STRLCPYO) $(XLIBS)

all-$(MODNAME): $(ROOTX)

clean-$(MODNAME):
		@rm -f $(ROOTXO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTXDEP) $(ROOTX)

distclean::     distclean-$(MODNAME)
