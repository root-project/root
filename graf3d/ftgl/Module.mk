# Module.mk for ftgl module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 15/11/2006

MODNAME      := ftgl
MODDIR       := $(ROOT_SRCDIR)/graf3d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FTGLDIR      := $(MODDIR)
FTGLDIRS     := $(FTGLDIR)/src
FTGLDIRI     := $(FTGLDIR)/inc

##### libFTGL #####
FTGLH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FTGLS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FTGLO        := $(call stripsrc,$(FTGLS:.cxx=.o))

FTGLDEP      := $(FTGLO:.o=.d)

FTGLLIB      := $(LPATH)/libFTGL.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FTGLH))
ALLLIBS     += $(FTGLLIB)

# include all dependency files
INCLUDEFILES += $(FTGLDEP)

ifeq ($(MACOSX_MINOR),3)
FTGLLIBEXTRA += -lz
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FTGLDIRI)/%.h
		cp $< $@

$(FTGLLIB):     $(FTGLO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) $(FTGLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFTGL.$(SOEXT) $@ \
		   "$(FTGLO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(FTGLLIBEXTRA) $(GLLIBS)"

all-$(MODNAME): $(FTGLLIB)

clean-$(MODNAME):
		@rm -f $(FTGLO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FTGLDEP) $(FTGLLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(FTGLO):     $(FREETYPEDEP)
$(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(OPENGLINCDIR:%=-I%)
