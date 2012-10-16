# Module.mk for globals module
# Copyright (c) 2012 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/10/2012

MODNAME      := globals
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLOBALSDIR   := $(MODDIR)
GLOBALSDIRS  := $(GLOBALSDIR)/src
GLOBALSDIRI  := $(GLOBALSDIR)/inc

##### libGlobals #####
GLOBALSS     := $(wildcard $(MODDIRS)/*.cxx)
GLOBALSO     := $(call stripsrc,$(GLOBALSS:.cxx=.o))

GLOBALSDEP   := $(GLOBALSO:.o=.d)

GLOBALSLIB   := $(LPATH)/libGlobals.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLOBALSH))
ALLLIBS     += $(GLOBALSLIB)

# include all dependency files
INCLUDEFILES += $(GLOBALSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GLOBALSDIRI)/%.h
		cp $< $@

$(GLOBALSLIB): $(GLOBALSO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGlobals.$(SOEXT) $@ \
		   "$(GLOBALSO)" "$(GLOBALSLIBEXTRA)"

all-$(MODNAME): $(GLOBALSLIB)

clean-$(MODNAME):
		@rm -f $(GLOBALSO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GLOBALSDEP) $(GLOBALSLIB)

distclean::     distclean-$(MODNAME)
