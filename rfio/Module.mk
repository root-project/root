# Module.mk for rfio module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := rfio
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RFIODIR      := $(MODDIR)
RFIODIRS     := $(RFIODIR)/src
RFIODIRI     := $(RFIODIR)/inc

##### libRFIO #####
RFIOL        := $(MODDIRI)/LinkDef.h
RFIODS       := $(MODDIRS)/G__RFIO.cxx
RFIODO       := $(RFIODS:.cxx=.o)
RFIODH       := $(RFIODS:.cxx=.h)

RFIOH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RFIOS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RFIOO        := $(RFIOS:.cxx=.o)

RFIODEP      := $(RFIOO:.o=.d) $(RFIODO:.o=.d)

RFIOLIB      := $(LPATH)/libRFIO.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RFIOH))
ALLLIBS     += $(RFIOLIB)

# include all dependency files
INCLUDEFILES += $(RFIODEP)

##### local rules #####
include/%.h:    $(RFIODIRI)/%.h
		cp $< $@

$(RFIOLIB):     $(RFIOO) $(RFIODO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRFIO.$(SOEXT) $@ "$(RFIOO) $(RFIODO)" \
		   "$(RFIOLIBEXTRA) $(RFIO)"

$(RFIODS):      $(RFIOH) $(RFIOL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@$(ROOTCINTTMP) -f $@ -c $(RFIOH) $(RFIOL)

$(RFIODO):      $(RFIODS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-rfio:       $(RFIOLIB)

clean-rfio:
		@rm -f $(RFIOO) $(RFIODO)

clean::         clean-rfio

distclean-rfio: clean-rfio
		@rm -f $(RFIODEP) $(RFIODS) $(RFIODH) $(RFIOLIB)

distclean::     distclean-rfio
