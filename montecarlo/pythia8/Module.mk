# Module.mk for pythia8 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun 01/11/2007

MODNAME      := pythia8
MODDIR       := $(ROOT_SRCDIR)/montecarlo/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PYTHIA8DIR   := $(MODDIR)
PYTHIA8DIRS  := $(PYTHIA8DIR)/src
PYTHIA8DIRI  := $(PYTHIA8DIR)/inc

##### libEGPythia8 #####
PYTHIA8L     := $(MODDIRI)/LinkDef.h
PYTHIA8DS    := $(call stripsrc,$(MODDIRS)/G__Pythia8.cxx)
PYTHIA8DO    := $(PYTHIA8DS:.cxx=.o)
PYTHIA8DH    := $(PYTHIA8DS:.cxx=.h)

PYTHIA8H     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PYTHIA8S     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PYTHIA8O     := $(call stripsrc,$(PYTHIA8S:.cxx=.o))

PYTHIA8DEP   := $(PYTHIA8O:.o=.d) $(PYTHIA8DO:.o=.d)

PYTHIA8LIB   := $(LPATH)/libEGPythia8.$(SOEXT)
PYTHIA8MAP   := $(PYTHIA8LIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYTHIA8H))
ALLLIBS     += $(PYTHIA8LIB)
ALLMAPS     += $(PYTHIA8MAP)

# include all dependency files
INCLUDEFILES += $(PYTHIA8DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PYTHIA8DIRI)/%.h
		cp $< $@

$(PYTHIA8LIB):  $(PYTHIA8O) $(PYTHIA8DO) $(ORDER_) $(MAINLIBS) $(PYTHIA8LIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEGPythia8.$(SOEXT) $@ \
		   "$(PYTHIA8O) $(PYTHIA8DO)" \
		   "$(PYTHIA8LIBEXTRA) $(FPYTHIA8LIBDIR) $(FPYTHIA8LIB)"

$(call pcmrule,PYTHIA8)
	$(noop)

$(PYTHIA8DS):   $(PYTHIA8H) $(PYTHIA8L) $(ROOTCLINGEXE) $(call pcmdep,PYTHIA8)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PYTHIA8) -c $(FPYTHIA8INCDIR:%=-I%) $(PYTHIA8H) $(PYTHIA8L)

$(PYTHIA8MAP):  $(PYTHIA8H) $(PYTHIA8L) $(ROOTCLINGEXE) $(call pcmdep,PYTHIA8)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(PYTHIA8DS) $(call dictModule,PYTHIA8) -c $(FPYTHIA8INCDIR:%=-I%) $(PYTHIA8H) $(PYTHIA8L)

all-$(MODNAME): $(PYTHIA8LIB)

clean-$(MODNAME):
		@rm -f $(PYTHIA8O) $(PYTHIA8DO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PYTHIA8DEP) $(PYTHIA8DS) $(PYTHIA8DH) \
		   $(PYTHIA8LIB) $(PYTHIA8MAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PYTHIA8O):    CXXFLAGS += $(FPYTHIA8INCDIR:%=-I%)
$(PYTHIA8DO):   CXXFLAGS += $(FPYTHIA8INCDIR:%=-I%)
