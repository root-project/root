# Module.mk for gfal module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 9/12/2005

MODDIR       := gfal
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GFALDIR      := $(MODDIR)
GFALDIRS     := $(GFALDIR)/src
GFALDIRI     := $(GFALDIR)/inc

##### libGFAL #####
GFALL        := $(MODDIRI)/LinkDef.h
GFALDS       := $(MODDIRS)/G__GFAL.cxx
GFALDO       := $(GFALDS:.cxx=.o)
GFALDH       := $(GFALDS:.cxx=.h)

GFALH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GFALS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GFALO        := $(GFALS:.cxx=.o)

GFALDEP      := $(GFALO:.o=.d) $(GFALDO:.o=.d)

GFALLIB      := $(LPATH)/libGFAL.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GFALH))
ALLLIBS     += $(GFALLIB)

# include all dependency files
INCLUDEFILES += $(GFALDEP)

##### local rules #####
include/%.h:    $(GFALDIRI)/%.h
		cp $< $@

$(GFALLIB):     $(GFALO) $(GFALDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGFAL.$(SOEXT) $@ "$(GFALO) $(GFALDO)" \
		   "$(GFALLIBEXTRA) $(GFALLIBDIR) $(GFALCLILIB)"

$(GFALDS):      $(GFALH) $(GFALL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GFALH) $(GFALL)

$(GFALDO):      $(GFALDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-gfal:       $(GFALLIB)

map-gfal:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GFALLIB) \
		   -d $(GFALLIBDEP) -c $(GFALL)

map::           map-gfal

clean-gfal:
		@rm -f $(GFALO) $(GFALDO)

clean::         clean-gfal

distclean-gfal: clean-gfal
		@rm -f $(GFALDEP) $(GFALDS) $(GFALDH) $(GFALLIB)

distclean::     distclean-gfal

##### extra rules ######
$(GFALO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(GFALINCDIR:%=-I%) -o $@ -c $<
