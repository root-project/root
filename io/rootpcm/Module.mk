# Module.mk for rootpcm module
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-19

MODNAME      := rootpcm
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTPCMDIR      := $(MODDIR)
ROOTPCMDIRS     := $(ROOTPCMDIR)/src
ROOTPCMDIRI     := $(ROOTPCMDIR)/inc
ROOTPCMDIRR     := $(ROOTPCMDIR)/res

##### $(ROOTPCMO) (part of libRIO) #####
ROOTPCMS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOTPCMO        := $(call stripsrc,$(ROOTPCMS:.cxx=.o))

ROOTPCMDEP      := $(ROOTPCMO:.o=.d) $(ROOTPCMDO:.o=.d)

# include all dependency files
INCLUDEFILES += $(ROOTPCMDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME): $(ROOTPCMO)
clean-$(MODNAME):
		@rm -f $(ROOTPCMO) $(GLINGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTPCMDEP)

distclean::     distclean-$(MODNAME)
