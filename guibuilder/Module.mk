# Module.mk for guibuilder module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Valeriy Onuchin, 19/8/2004

MODDIR       := guibuilder
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GUIBLDDIR    := $(MODDIR)
GUIBLDDIRS   := $(GUIBLDDIR)/src
GUIBLDDIRI   := $(GUIBLDDIR)/inc

##### libGuiBld #####
GUIBLDL      := $(MODDIRI)/LinkDef.h
GUIBLDDS     := $(MODDIRS)/G__GuiBld.cxx
GUIBLDDO     := $(GUIBLDDS:.cxx=.o)
GUIBLDDH     := $(GUIBLDDS:.cxx=.h)

GUIBLDH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GUIBLDS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GUIBLDO      := $(GUIBLDS:.cxx=.o)

GUIBLDDEP    := $(GUIBLDO:.o=.d) $(GUIBLDDO:.o=.d)

GUIBLDLIB    := $(LPATH)/libGuiBld.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GUIBLDH))
ALLLIBS     += $(GUIBLDLIB)

# include all dependency files
INCLUDEFILES += $(GUIBLDDEP)

##### local rules #####
include/%.h:    $(GUIBLDDIRI)/%.h
		cp $< $@

$(GUIBLDLIB):   $(GUIBLDO) $(GUIBLDDO) $(MAINLIBS) $(GUIBLDLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGuiBld.$(SOEXT) $@ "$(GUIBLDO) $(GUIBLDDO)" \
		   "$(GUIBLDLIBEXTRA)"

$(GUIBLDDS):    $(GUIBLDH) $(GUIBLDL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GUIBLDH) $(GUIBLDL)

$(GUIBLDDO):    $(GUIBLDDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-guibuilder: $(GUIBLDLIB)

map-guibuilder: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GUIBLDLIB) \
		   -d $(GUIBLDLIBDEP) -c $(GUIBLDL)

map::           map-guibuilder

clean-guibuilder:
		@rm -f $(GUIBLDO) $(GUIBLDDO)

clean::         clean-guibuilder

distclean-guibuilder: clean-guibuilder
		@rm -f $(GUIBLDDEP) $(GUIBLDDS) $(GUIBLDDH) $(GUIBLDLIB)

distclean::     distclean-guibuilder
