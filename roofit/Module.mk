# ROOT Module.mk for roofitcore module
# Copyright (c) 2005 Wouter Verkerke
#
# Author: Wouter Verkerke, 18/4/2005

MODDIR       := roofit
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOFITDIR    := $(MODDIR)
ROOFITDIRS   := $(ROOFITDIR)/src
ROOFITDIRI   := $(ROOFITDIR)/inc

##### libRooFit #####
ROOFITL      := $(MODDIRI)/LinkDef1.h
ROOFITDS     := $(MODDIRS)/G__RooFit.cxx
ROOFITDO     := $(ROOFITDS:.cxx=.o)
ROOFITDH     := $(ROOFITDS:.cxx=.h)

ROOFITH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ROOFITS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOFITO      := $(ROOFITS:.cxx=.o)

ROOFITDEP    := $(ROOFITO:.o=.d) $(ROOFITDO:.o=.d)

ROOFITLIB    := $(LPATH)/libRooFit.$(SOEXT)
ROOFITMAP    := $(ROOFITLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOFITH))
ALLLIBS      += $(ROOFITLIB)
ALLMAPS      += $(ROOFITMAP)

# include all dependency files
INCLUDEFILES += $(ROOFITDEP)

##### local rules #####
include/%.h:    $(ROOFITDIRI)/%.h
		cp $< $@

$(ROOFITLIB):   $(ROOFITO) $(ROOFITDO) $(ORDER_) $(MAINLIBS) $(ROOFITLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooFit.$(SOEXT) $@ "$(ROOFITO) $(ROOFITDO)" \
		   "$(ROOFITLIBEXTRA)"

$(ROOFITDS):    $(ROOFITH) $(ROOFITL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITH) $(ROOFITL)

$(ROOFITMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(ROOFITL)
		$(RLIBMAP) -o $(ROOFITMAP) -l $(ROOFITLIB) \
		   -d $(ROOFITLIBDEPM) -c $(ROOFITL)

all-roofit:     $(ROOFITLIB) $(ROOFITMAP)

clean-roofit:
		@rm -f $(ROOFITO) $(ROOFITDO)

clean::         clean-roofit

distclean-roofit: clean-roofit
		@rm -rf $(ROOFITDEP) $(ROOFITLIB) $(ROOFITMAP) \
		   $(ROOFITDS) $(ROOFITDH)

distclean::     distclean-roofit
