# Module.mk for winnt module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := winnt
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

WINNTH1      := $(MODDIRI)/TWinNTSystem.h $(MODDIRI)/TWinNTInput.h
WINNTH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WINNTS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WINNTO       := $(WINNTS:.cxx=.o)

WINNTDEP     := $(WINNTO:.o=.d) $(WINNTDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WINNTH))

# include all dependency files
INCLUDEFILES += $(WINNTDEP)

##### local rules #####
include/%.h:    $(WINNTDIRI)/%.h
		cp $< $@

$(WINNTDS):     $(WINNTH1) $(WINNTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WINNTH1) $(WINNTL)

$(WINNTDO):     $(WINNTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-winnt:      $(WINNTO) $(WINNTDO)

clean-winnt:
		@rm -f $(WINNTO) $(WINNTDO)

clean::         clean-winnt

distclean-winnt: clean-winnt
		@rm -f $(WINNTDEP) $(WINNTDS) $(WINNTDH)

distclean::     distclean-winnt
