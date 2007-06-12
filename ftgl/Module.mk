# Module.mk for ftgl module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 15/11/2006

MODDIR       := ftgl
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FTGLDIR      := $(MODDIR)
FTGLDIRS     := $(FTGLDIR)/src
FTGLDIRI     := $(FTGLDIR)/inc

##### libFTGL #####
FTGLH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FTGLS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FTGLO        := $(FTGLS:.cxx=.o)

FTGLDEP      := $(FTGLO:.o=.d)

FTGLLIB      := $(LPATH)/libFTGL.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FTGLH))
ALLLIBS     += $(FTGLLIB)

# include all dependency files
INCLUDEFILES += $(FTGLDEP)

ifeq ($(ARCH),win32)
GLLIBS       := opengl32.lib glu32.lib
endif
ifeq ($(MACOSX_MINOR),3)
FTGLLIBEXTRA += -lz
endif

##### local rules #####
include/%.h:    $(FTGLDIRI)/%.h
		cp $< $@

$(FTGLLIB):     $(FTGLO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) $(FTGLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFTGL.$(SOEXT) $@ \
		   "$(FTGLO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(FTGLLIBEXTRA) $(XLIBS) $(GLLIBS)"

all-ftgl:     $(FTGLLIB)

clean-ftgl:
		@rm -f $(FTGLO)

clean::         clean-ftgl

distclean-ftgl: clean-ftgl
		@rm -f $(FTGLDEP) $(FTGLLIB)

distclean::     distclean-ftgl

##### extra rules ######
$(FTGLO):     $(FREETYPEDEP)
$(FTGLO):     CXXFLAGS += $(FREETYPEINC)


