# Module.mk for table module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := table
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TABLEDIR     := $(MODDIR)
TABLEDIRS    := $(TABLEDIR)/src
TABLEDIRI    := $(TABLEDIR)/inc

##### libTable #####
TABLEL       := $(MODDIRI)/LinkDef.h
TABLEDS      := $(MODDIRS)/G__Table.cxx
TABLEDO      := $(TABLEDS:.cxx=.o)
TABLEDH      := $(TABLEDS:.cxx=.h)

TABLEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TABLES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TABLEO       := $(TABLES:.cxx=.o)

TABLEDEP     := $(TABLEO:.o=.d) $(TABLEDO:.o=.d)

TABLELIB     := $(LPATH)/libTable.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TABLEH))
ALLLIBS     += $(TABLELIB)

# include all dependency files
INCLUDEFILES += $(TABLEDEP)

##### local rules #####
include/%.h:    $(TABLEDIRI)/%.h
		cp $< $@

$(TABLELIB):    $(TABLEO) $(TABLEDO) $(MAINLIBS) $(TABLELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTable.$(SOEXT) $@ "$(TABLEO) $(TABLEDO)" \
		   "$(TABLELIBEXTRA)"

$(TABLEDS):     $(TABLEH) $(TABLEL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TABLEH) $(TABLEL)

$(TABLEDO):     $(TABLEDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-table:      $(TABLELIB)

map-table:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(TABLELIB) \
		   -d $(TABLELIBDEP) -c $(TABLEL)

map::           map-table

clean-table:
		@rm -f $(TABLEO) $(TABLEDO)

clean::         clean-table

distclean-table: clean-table
		@rm -f $(TABLEDEP) $(TABLEDS) $(TABLEDH) $(TABLELIB)

distclean::     distclean-table
