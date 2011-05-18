# Module.mk for new module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := newdelete
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NEWDIR       := $(MODDIR)
NEWDIRS      := $(NEWDIR)/src
NEWDIRI      := $(NEWDIR)/inc

##### libNew #####
NEWH         := $(wildcard $(MODDIRI)/*.h)
NEWS         := $(wildcard $(MODDIRS)/*.cxx)
NEWO         := $(call stripsrc,$(NEWS:.cxx=.o))

NEWDEP       := $(NEWO:.o=.d)

NEWLIB       := $(LPATH)/libNew.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NEWH))
ALLLIBS     += $(NEWLIB)

# include all dependency files
INCLUDEFILES += $(NEWDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(NEWDIRI)/%.h
		cp $< $@

$(NEWLIB):      $(NEWO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNew.$(SOEXT) $@ "$(NEWO)" "$(NEWLIBEXTRA)"

all-$(MODNAME): $(NEWLIB)

clean-$(MODNAME):
		@rm -f $(NEWO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(NEWDEP) $(NEWLIB)

distclean::     distclean-$(MODNAME)
