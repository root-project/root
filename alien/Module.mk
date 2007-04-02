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
ALIENMAP     := $(ALIENLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ALIENH))
ALLLIBS     += $(ALIENLIB)
ALLMAPS     += $(ALIENMAP)

# include all dependency files
INCLUDEFILES += $(ALIENDEP)

##### local rules #####
include/%.h:    $(ALIENDIRI)/%.h
		cp $< $@

$(ALIENLIB):    $(ALIENO) $(ALIENDO) $(ORDER_) $(MAINLIBS) $(ALIENLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRAliEn.$(SOEXT) $@ "$(ALIENO) $(ALIENDO)" \
		   "$(ALIENLIBEXTRA) $(ALIENLIBDIR) $(ALIENCLILIB)"

$(ALIENDS):     $(ALIENH) $(ALIENL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ALIENH) $(ALIENL)

$(ALIENMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(ALIENL)
		$(RLIBMAP) -o $(ALIENMAP) -l $(ALIENLIB) \
		   -d $(ALIENLIBDEPM) -c $(ALIENL)

all-alien:      $(ALIENLIB) $(ALIENMAP)

clean-alien:
		@rm -f $(ALIENO) $(ALIENDO)

clean::         clean-alien

distclean-alien: clean-alien
		@rm -f $(ALIENDEP) $(ALIENDS) $(ALIENDH) $(ALIENLIB) $(ALIENMAP)

distclean::     distclean-alien

##### extra rules ######
$(ALIENO) $(ALIENDO): CXXFLAGS += $(ALIENINCDIR:%=-I%)
