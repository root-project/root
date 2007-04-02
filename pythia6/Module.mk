# Module.mk for pythia6 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := pythia6
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PYTHIA6DIR   := $(MODDIR)
PYTHIA6DIRS  := $(PYTHIA6DIR)/src
PYTHIA6DIRI  := $(PYTHIA6DIR)/inc

##### libEGPythia6 #####
PYTHIA6L     := $(MODDIRI)/LinkDef.h
PYTHIA6DS    := $(MODDIRS)/G__Pythia6.cxx
PYTHIA6DO    := $(PYTHIA6DS:.cxx=.o)
PYTHIA6DH    := $(PYTHIA6DS:.cxx=.h)

PYTHIA6H     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PYTHIA6S     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PYTHIA6O     := $(PYTHIA6S:.cxx=.o)

PYTHIA6DEP   := $(PYTHIA6O:.o=.d) $(PYTHIA6DO:.o=.d)

PYTHIA6LIB   := $(LPATH)/libEGPythia6.$(SOEXT)
PYTHIA6MAP   := $(PYTHIA6LIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYTHIA6H))
ALLLIBS     += $(PYTHIA6LIB)
ALLMAPS     += $(PYTHIA6MAP)

# include all dependency files
INCLUDEFILES += $(PYTHIA6DEP)

##### local rules #####
include/%.h:    $(PYTHIA6DIRI)/%.h
		cp $< $@

$(PYTHIA6LIB):  $(PYTHIA6O) $(PYTHIA6DO) $(ORDER_) $(MAINLIBS) $(PYTHIA6LIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEGPythia6.$(SOEXT) $@ \
		   "$(PYTHIA6O) $(PYTHIA6DO)" \
		   "$(PYTHIA6LIBEXTRA) $(FPYTHIA6LIBDIR) $(FPYTHIA6LIB)"

$(PYTHIA6DS):   $(PYTHIA6H) $(PYTHIA6L) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PYTHIA6H) $(PYTHIA6L)

$(PYTHIA6MAP):  $(RLIBMAP) $(MAKEFILEDEP) $(PYTHIA6L)
		$(RLIBMAP) -o $(PYTHIA6MAP) -l $(PYTHIA6LIB) \
		   -d $(PYTHIA6LIBDEPM) -c $(PYTHIA6L)

all-pythia6:    $(PYTHIA6LIB) $(PYTHIA6MAP)

clean-pythia6:
		@rm -f $(PYTHIA6O) $(PYTHIA6DO)

clean::         clean-pythia6

distclean-pythia6: clean-pythia6
		@rm -f $(PYTHIA6DEP) $(PYTHIA6DS) $(PYTHIA6DH) $(PYTHIA6LIB) $(PYTHIA6MAP)

distclean::     distclean-pythia6

##### extra rules ######
$(PYTHIA6O):    CXXFLAGS += $(FPYTHIA6CPPFLAGS)
