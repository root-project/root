# Module.mk for proofd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := proofd
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFDDIR    := $(MODDIR)
PROOFDDIRS   := $(PROOFDDIR)/src
PROOFDDIRI   := $(PROOFDDIR)/inc

##### proofd #####
PROOFDEXEH   := $(wildcard $(MODDIRI)/*.h)
PROOFDEXES   := $(wildcard $(MODDIRS)/*.cxx)
PROOFDEXEO   := $(PROOFDEXES:.cxx=.o)
PROOFDDEP    := $(PROOFDEXEO:.o=.d)
PROOFDEXE    := bin/proofd

##### authentication flags and libs defined in rootd/Module.mk #####

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFDEXEH))
ALLEXECS     += $(PROOFDEXE)

# include all dependency files
INCLUDEFILES += $(PROOFDDEP)

##### local rules #####
include/%.h:    $(PROOFDDIRI)/%.h
		cp $< $@

$(PROOFDEXE):   $(PROOFDEXEO)
		$(LD) $(LDFLAGS) -o $@ $(PROOFDEXEO) $(AUTHLIBS) $(CRYPTLIBS) \
		   $(SYSLIBS)

all-proofd:     $(PROOFDEXE)

clean-proofd:
		@rm -f $(PROOFDEXEO)

clean::         clean-proofd

distclean-proofd: clean-proofd
		@rm -f $(PROOFDDEP) $(PROOFDEXE)

distclean::     distclean-proofd

##### extra rules ######
$(PROOFDDIRS)/proofd.o: $(PROOFDDIRS)/proofd.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(AUTHFLAGS) -o $@ -c $<
