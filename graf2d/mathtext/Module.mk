# Module.mk for mathtex module
# Copyright (c) 2012 Rene Brun and Fons Rademakers
#
# Author: Olivier Couet, 30/10/2012

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

MATHTEXTDEP  := $(MATHTEXTO:.o=.d)

ifeq ($(PLATFORM),win32)
MATHTEXTLIB  := $(LPATH)/libmathtext.lib
MATHTEXTAR   := link.exe -lib /out:
else
MATHTEXTLIB  := $(LPATH)/libmathtext.a
MATHTEXTAR   := $(AR) cru 
endif

MATHTEXTLIBDEP := $(MATHTEXTLIB)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MATHTEXTH))
ALLLIBS     += $(MATHTEXTLIB)

# include all dependency files
INCLUDEFILES += $(MATHTEXTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MATHTEXTDIRI)/%.h
		cp $< $@

$(MATHTEXTLIB): $(MATHTEXTO)
		$(MATHTEXTAR)$@ $(MATHTEXTO)
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

all-$(MODNAME): $(MATHTEXTLIB)

clean-$(MODNAME):
		@rm -f $(MATHTEXTO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MATHTEXTDEP) $(MATHTEXTLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
