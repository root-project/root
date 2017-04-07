# Module.mk for cont module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := cont
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CONTDIR      := $(MODDIR)
CONTDIRS     := $(CONTDIR)/src
CONTDIRI     := $(CONTDIR)/inc

##### libCont (part of libCore) #####
CONTL        := $(MODDIRI)/LinkDef.h

CONTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CONTH        += $(wildcard $(MODDIRI)/ROOT/*.hxx)
CONTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CONTO        := $(call stripsrc,$(CONTS:.cxx=.o))

CONTDEP      := $(CONTO:.o=.d) $(CONTDO:.o=.d)

# used in the main Makefile
CONTH_REL   := $(patsubst $(MODDIRI)/%,include/%,$(CONTH))
ALLHDRS     += $(CONTH_REL)

# include all dependency files
INCLUDEFILES += $(CONTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:	$(CONTDIRI)/%.h
		cp $< $@

include/%.hxx:	$(CONTDIRI)/%.hxx
		mkdir -p include/ROOT;
		cp $< $@

all-$(MODNAME): $(CONTO)

clean-$(MODNAME):
		@rm -f $(CONTO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CONTDEP)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(COREDO): NOOPT = $(OPT)
