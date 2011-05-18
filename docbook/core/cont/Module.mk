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
CONTDS       := $(call stripsrc,$(MODDIRS)/G__Cont.cxx)
CONTDO       := $(CONTDS:.cxx=.o)
CONTDH       := $(CONTDS:.cxx=.h)

CONTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CONTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CONTO        := $(call stripsrc,$(CONTS:.cxx=.o))

CONTDEP      := $(CONTO:.o=.d) $(CONTDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CONTH))

# include all dependency files
INCLUDEFILES += $(CONTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CONTDIRI)/%.h
		cp $< $@

$(CONTDS):      $(CONTH) $(CONTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CONTH) $(CONTL)

all-$(MODNAME): $(CONTO) $(CONTDO)

clean-$(MODNAME):
		@rm -f $(CONTO) $(CONTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CONTDEP) $(CONTDS) $(CONTDH)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(CONTDO): NOOPT = $(OPT)
