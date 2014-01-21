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
# ALLHDRS     +=
ALLLIBS      += $(FTGLLIB)

# include all dependency files
INCLUDEFILES += $(FTGLDEP)

FTGLINC		 := -I$(MODDIRI)

ifeq ($(MACOSX_MINOR),3)
FTGLLIBEXTRA += -lz
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

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
$(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(FTGLINC) $(OPENGLINCDIR:%=-I%)
ifeq ($(MACOSX_GLU_DEPRECATED),yes)
$(FTGLO):     CXXFLAGS += -Wno-deprecated-declarations
endif
