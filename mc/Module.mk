# Module.mk for mc module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/4/2002

MODDIR       := mc
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MCDIR        := $(MODDIR)
MCDIRS       := $(MCDIR)/src
MCDIRI       := $(MCDIR)/inc

##### libMC #####
MCL          := $(MODDIRI)/LinkDef.h
MCDS         := $(MODDIRS)/G__MC.cxx
MCDO         := $(MCDS:.cxx=.o)
MCDH         := $(MCDS:.cxx=.h)

MCH1         := $(wildcard $(MODDIRI)/T*.h)
MCH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MCS          := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MCO          := $(MCS:.cxx=.o)

MCDEP        := $(MCO:.o=.d) $(MCDO:.o=.d)

MCLIB        := $(LPATH)/libMC.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MCH))
ALLLIBS     += $(MCLIB)

# include all dependency files
INCLUDEFILES += $(MCDEP)

##### local rules #####
include/%.h:    $(MCDIRI)/%.h
		cp $< $@

$(MCLIB):       $(MCO) $(MCDO) $(MAINLIBS) $(MCLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMC.$(SOEXT) $@ "$(MCO) $(MCDO)" \
		   "$(MCLIBEXTRA)"

$(MCDS):        $(MCH1) $(MCL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MCH1) $(MCL)

$(MCDO):        $(MCDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-mc:         $(MCLIB)

clean-mc:
		@rm -f $(MCO) $(MCDO)

clean::         clean-mc

distclean-mc:   clean-mc
		@rm -f $(MCDEP) $(MCDS) $(MCDH) $(MCLIB)

distclean::     distclean-mc
