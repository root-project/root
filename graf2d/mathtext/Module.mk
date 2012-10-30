# Module.mk for mathtex module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: 

MODNAME      := mathtext
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHTEXTDIR  := $(MODDIR)
MATHTEXTDIRS := $(MATHTEXTDIR)/src
MATHTEXTDIRI := $(MATHTEXTDIR)/inc

##### libmathtext #####
MATHTEXTH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATHTEXTS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHTEXTO    := $(call stripsrc,$(MATHTEXTS:.cxx=.o))

MATHTEXTDEP  := $(MATHTEXTO:.o=.d) $(MATHTEXTDO:.o=.d)

ifeq ($(PLATFORM),win32)
MATHTEXTLIB  := $(LPATH)\libmathtext.lib
ARCHIVE      := link.exe -lib /out:
else
MATHTEXTLIB  := $(LPATH)/libmathtext.a
ARCHIVE      := $(AR) cru 
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MATHTEXTH))
ALLLIBS     += $(MATHTEXTLIB)

# include all dependency files
INCLUDEFILES += $(MATHTEXTDEP)

MATHTEXTINC  := $(MATHTEXTDIRI:%=-I%)
MATHTEXTDEP  := $(MATHTEXTLIB)
MATHTEXTLDFLAGS :=

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MATHTEXTDIRI)/%.h
		cp $< $@

$(MATHTEXTLIB): $(MATHTEXTO) $(ORDER_)
		$(ARCHIVE)$@ $(MATHTEXTO)

all-$(MODNAME): $(MATHTEXTLIB)

clean-$(MODNAME):
		@rm -f $(MATHTEXTO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MATHTEXTDEP) $(MATHTEXTDS) $(MATHTEXTDH) \
	$(MATHTEXTLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
