# Module.mk for alien module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/5/2002

MODDIR       := alien
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ALIENDIR     := $(MODDIR)
ALIENDIRS    := $(ALIENDIR)/src
ALIENDIRI    := $(ALIENDIR)/inc

##### libRAliEn #####
ALIENL       := $(MODDIRI)/LinkDef.h
ALIENDS      := $(MODDIRS)/G__Alien.cxx
ALIENDO      := $(ALIENDS:.cxx=.o)
ALIENDH      := $(ALIENDS:.cxx=.h)

ALIENH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ALIENS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ALIENO       := $(ALIENS:.cxx=.o)

ALIENDEP     := $(ALIENO:.o=.d) $(ALIENDO:.o=.d)

ALIENLIB     := $(LPATH)/libRAliEn.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ALIENH))
ALLLIBS     += $(ALIENLIB)

# include all dependency files
INCLUDEFILES += $(ALIENDEP)

##### local rules #####
include/%.h:    $(ALIENDIRI)/%.h
		cp $< $@

$(ALIENLIB):    $(ALIENO) $(ALIENDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRAliEn.$(SOEXT) $@ "$(ALIENO) $(ALIENDO)" \
		   "$(ALIENLIBEXTRA) $(ALIENLIBDIR) $(ALIENCLILIB)"

$(ALIENDS):     $(ALIENH) $(ALIENL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ALIENH) $(ALIENL)

$(ALIENDO):     $(ALIENDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I$(ALIENINCDIR) -I. -o $@ -c $<

all-alien:      $(ALIENLIB)

clean-alien:
		@rm -f $(ALIENO) $(ALIENDO)

clean::         clean-alien

distclean-alien: clean-alien
		@rm -f $(ALIENDEP) $(ALIENDS) $(ALIENDH) $(ALIENLIB)

distclean::     distclean-alien

##### extra rules ######
$(ALIENO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(ALIENINCDIR) -o $@ -c $<
