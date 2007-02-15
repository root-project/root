# Module.mk for io module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Rene Brun 06/02/2007

MODDIR       := io
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

IODIR        := $(MODDIR)
IODIRS       := $(IODIR)/src
IODIRI       := $(IODIR)/inc

##### libRIO #####
IOL          := $(MODDIRI)/LinkDef.h
IODS         := $(MODDIRS)/G__IO.cxx
IODO         := $(IODS:.cxx=.o)
IODH         := $(IODS:.cxx=.h)

IOH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
IOS          := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
IOO          := $(IOS:.cxx=.o)

IODEP        := $(IOO:.o=.d) $(IODO:.o=.d)

IOLIB        := $(LPATH)/libRIO.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(IOH))
ALLLIBS      += $(IOLIB)

# include all dependency files
INCLUDEFILES += $(IODEP)

##### local rules #####
include/%.h:    $(IODIRI)/%.h
		cp $< $@

$(IOLIB):       $(IOO) $(IODO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRIO.$(SOEXT) $@ "$(IOO) $(IODO)" \
		   "$(IOLIBEXTRA)"

$(IODS):        $(IOH) $(IOL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(IOH) $(IOL)

all-io:         $(IOLIB)
#all-io:         $(IOO) $(IODO)

map-io:         $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(IOLIB) \
		   -d $(IOLIBDEP) -c $(IOL)

map::           map-io

clean-io:
		@rm -f $(IOO) $(IODO)

clean::         clean-io

distclean-io:   clean-io
		@rm -f $(IODEP) $(IODS) $(IODH)
#		@rm -f $(IODEP) $(IODS) $(IODH) $(IOLIB)

distclean::     distclean-io
