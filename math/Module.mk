# Module.mk for math module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Rene Brun 06/02/2007

MODDIR       := math
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHDIR      := $(MODDIR)
MATHDIRS     := $(MATHDIR)/src
MATHDIRI     := $(MATHDIR)/inc

##### libMath  #####
MATHL        := $(MODDIRI)/LinkDef.h
MATHDS       := $(MODDIRS)/G__Math.cxx
MATHDO       := $(MATHDS:.cxx=.o)
MATHDH       := $(MATHDS:.cxx=.h)

MATHH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATHS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHO        := $(MATHS:.cxx=.o)

MATHDEP      := $(MATHO:.o=.d) $(MATHDO:.o=.d)

MATHLIB      := $(LPATH)/libRMath.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MATHH))
#ALLLIBS     += $(MATHLIB)

# include all dependency files
INCLUDEFILES += $(MATHDEP)

##### local rules #####
include/%.h:    $(MATHDIRI)/%.h
		cp $< $@

$(MATHLIB):     $(MATHO) $(MATHDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRMath.$(SOEXT) $@ "$(MATHO) $(MATHDO)" \
		   "$(MATHLIBEXTRA)"

$(MATHDS):      $(MATHH) $(MATHL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MATHH) $(MATHL)

#all-math:       $(MATHLIB)
all-math:       $(MATHO) $(MATHDO)

#map-math:       $(RLIBMAP)
#		$(RLIBMAP) -r $(ROOTMAP) -l $(MATHLIB) \
#		   -d $(MATHLIBDEP) -c $(MATHL)

#map::           map-tree

clean-math:
		@rm -f $(MATHO) $(MATHDO)

clean::         clean-math

distclean-math: clean-math
		@rm -f $(MATHDEP) $(MATHDS) $(MATHDH)

distclean::     distclean-math
