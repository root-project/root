# Module.mk for rint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := rint
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RINTDIR      := $(MODDIR)
RINTDIRS     := $(RINTDIR)/src
RINTDIRI     := $(RINTDIR)/inc

##### libRint #####
RINTL        := $(MODDIRI)/LinkDef.h
RINTDS       := $(MODDIRS)/G__Rint.cxx
RINTDO       := $(RINTDS:.cxx=.o)
RINTDH       := $(RINTDS:.cxx=.h)

RINTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RINTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RINTO        := $(RINTS:.cxx=.o)

RINTDEP      := $(RINTO:.o=.d) $(RINTDO:.o=.d)

RINTLIB      := $(LPATH)/libRint.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RINTH))
ALLLIBS     += $(RINTLIB)

# include all dependency files
INCLUDEFILES += $(RINTDEP)

##### local rules #####
include/%.h:    $(RINTDIRI)/%.h
		cp $< $@

$(RINTLIB):     $(RINTO) $(RINTDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRint.$(SOEXT) $@ "$(RINTO) $(RINTDO)" \
		   "$(RINTLIBEXTRA)"

$(RINTDS):      $(RINTH) $(RINTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RINTH) $(RINTL)

$(RINTDO):      $(RINTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-rint:       $(RINTLIB)

map-rint:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(RINTLIB) \
		   -d $(RINTLIBDEP) -c $(RINTL)

map::           map-rint

clean-rint:
		@rm -f $(RINTO) $(RINTDO)

clean::         clean-rint

distclean-rint: clean-rint
		@rm -f $(RINTDEP) $(RINTDS) $(RINTDH) $(RINTLIB)

distclean::     distclean-rint
