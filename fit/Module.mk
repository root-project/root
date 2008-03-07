# Module.mk for fit module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 07/05/2003

MODDIR        := fit
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

FITDIR    := $(MODDIR)
FITDIRS   := $(FITDIR)/src
FITDIRI   := $(FITDIR)/inc

##### libFit #####
FITL     := $(MODDIRI)/LinkDef.h
FITDS    := $(MODDIRS)/G__Fit.cxx
FITDO    := $(FITDS:.cxx=.o)
FITDH    := $(FITDS:.cxx=.h)

#FITAH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FITH    := $(filter-out $(MODDIRI)/Fit/Chi2Grad%,$(wildcard $(MODDIRI)/Fit/*.h))
#FITBH    := $(MODDIR)/../mathmore/inc/Math/WrappedTF1.h \
#	    $(MODDIR)/../mathmore/inc/Math/WrappedMultiTF1.h  
# FITBH    := $(MODDIRI)/Fit/BinPoint.h \
# 	    $(MODDIRI)/Fit/DataRange.h \
# 	    $(MODDIRI)/Fit/DataOptions.h \
# 	    $(MODDIRI)/Fit/DataVector.h  \
# 	    $(MODDIRI)/Fit/WrappedTF1.h   

#FITH     := $(FITAH) $(FITBH)
FITS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FITO     := $(FITS:.cxx=.o)

FITDEP   := $(FITO:.o=.d) $(FITDO:.o=.d)

FITLIB   := $(LPATH)/libFit.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FITH))
#ALLLIBS      += $(FITLIB)

# include all dependency files
INCLUDEFILES += $(FITDEP)

##### local rules #####
include/Fit/%.h: $(FITDIRI)/Fit/%.h
		@(if [ ! -d "include/Fit" ]; then     \
		   mkdir -p include/Fit;              \
		fi)
		cp $< $@

include/%.h:    $(FITDIRI)/%.h
		cp $< $@


$(FITLIB):  $(FITO) $(FITDO) $(ORDER_) $(MAINLIBS) $(FITLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFit.$(SOEXT) $@ \
		   "$(FITO) $(FITDO)" \
		   "$(FITLIBEXTRA)"

$(FITDS):   $(FITH) $(FITL) $(ROOTCINTTMPEXE) 
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FITH) $(FITL)

all-fit:    $(FITO) $(FITDO) 

$(FITMAP):    	$(RLIBMAP) $(MAKEFILEDEP) $(FITL)
	 	$(RLIBMAP) -o $(FITMAP) -l $(FITLIB) \
		   -d $(FITLIBDEPM) -c $(FITL)

map::           map-fit

test-fit: 	$(FITLIB)
		cd $(FITDIR)/test; make

clean-fit:
		@rm -f $(FITO) $(FITDO)

clean::         clean-fit

distclean-fit: clean-fit
		@rm -f $(FITDEP) $(FITDS) $(FITDH) $(FITLIB)
		@rm -rf include/Fit

distclean::     distclean-fit
