# Module.mk for cont module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := cont
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CONTDIR      := $(MODDIR)
CONTDIRS     := $(CONTDIR)/src
CONTDIRI     := $(CONTDIR)/inc

##### libCont (part of libCore) #####
CONTL        := $(MODDIRI)/LinkDef.h
CONTDS       := $(MODDIRS)/G__Cont.cxx
CONTDO       := $(CONTDS:.cxx=.o)
CONTDH       := $(CONTDS:.cxx=.h)

CONTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CONTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CONTO        := $(CONTS:.cxx=.o)

CONTDEP      := $(CONTO:.o=.d) $(CONTDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CONTH))

# include all dependency files
INCLUDEFILES += $(CONTDEP)

##### local rules #####
include/%.h:    $(CONTDIRI)/%.h
		cp $< $@

$(CONTDS):      $(CONTH) $(CONTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CONTH) $(CONTL)

$(CONTDO):      $(CONTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-cont:       $(CONTO) $(CONTDO)

clean-cont:
		@rm -f $(CONTO) $(CONTDO)

clean::         clean-cont

distclean-cont: clean-cont
		@rm -f $(CONTDEP) $(CONTDS) $(CONTDH)

distclean::     distclean-cont
