# Module.mk for winnt module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := winnt
MODDIR       := core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WINNTDIR     := $(MODDIR)
WINNTDIRS    := $(WINNTDIR)/src
WINNTDIRI    := $(WINNTDIR)/inc

##### libWinNT (part of libCore) #####
WINNTL       := $(MODDIRI)/LinkDef.h
WINNTDS      := $(MODDIRS)/G__WinNT.cxx
WINNTDO      := $(WINNTDS:.cxx=.o)
WINNTDH      := $(WINNTDS:.cxx=.h)

WINNTH1      := $(MODDIRI)/TWinNTSystem.h
WINNTH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WINNTS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WINNTO       := $(WINNTS:.cxx=.o)

WINNTDEP     := $(WINNTO:.o=.d) $(WINNTDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WINNTH))

# include all dependency files
INCLUDEFILES += $(WINNTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(WINNTDIRI)/%.h
		cp $< $@

$(WINNTDS):     $(WINNTH1) $(WINNTL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WINNTH1) $(WINNTL)

all-$(MODNAME): $(WINNTO) $(WINNTDO)

clean-$(MODNAME):
		@rm -f $(WINNTO) $(WINNTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(WINNTDEP) $(WINNTDS) $(WINNTDH)

distclean::     distclean-$(MODNAME)
