# Module.mk for new module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := newdelete
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NEWDIR       := $(MODDIR)
NEWDIRS      := $(NEWDIR)/src
NEWDIRI      := $(NEWDIR)/inc

##### libNew #####
NEWH         := $(wildcard $(MODDIRI)/*.h)
NEWS         := $(wildcard $(MODDIRS)/*.cxx)
NEWO         := $(NEWS:.cxx=.o)

NEWDEP       := $(NEWO:.o=.d)

NEWLIB       := $(LPATH)/libNew.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NEWH))
ALLLIBS     += $(NEWLIB)

# include all dependency files
INCLUDEFILES += $(NEWDEP)

##### local rules #####
include/%.h:    $(NEWDIRI)/%.h
		cp $< $@

$(NEWLIB):      $(NEWO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNew.$(SOEXT) $@ "$(NEWO)" "$(NEWLIBEXTRA)"

all-new:        $(NEWLIB)

clean-new:
		@rm -f $(NEWO)

clean::         clean-new

distclean-new:  clean-new
		@rm -f $(NEWDEP) $(NEWLIB)

distclean::     distclean-new
