# Module.mk for rootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := rootd
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTDDIR     := $(MODDIR)
ROOTDDIRS    := $(ROOTDDIR)/src
ROOTDDIRI    := $(ROOTDDIR)/inc

##### rootd #####
ROOTDH       := $(wildcard $(MODDIRI)/*.h)
ROOTDS       := $(wildcard $(MODDIRS)/*.cxx)
ROOTDO       := $(ROOTDS:.cxx=.o)
ROOTDDEP     := $(ROOTDO:.o=.d)
ROOTD        := bin/rootd

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTDH))
ALLEXECS     += $(ROOTD)

# include all dependency files
INCLUDEFILES += $(ROOTDDEP)

##### local rules #####
include/%.h:    $(ROOTDDIRI)/%.h
		cp $< $@

$(ROOTD):       $(ROOTDO) $(RSAO) $(SNPRINTFO) $(GLBPATCHO) $(RPDUTILO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTDO) $(RPDUTILO) $(GLBPATCHO) \
		   $(RSAO) $(SNPRINTFO) $(CRYPTLIBS) $(AUTHLIBS) $(SYSLIBS)

all-rootd:      $(ROOTD)

clean-rootd:
		@rm -f $(ROOTDO)

clean::         clean-rootd

distclean-rootd: clean-rootd
		@rm -f $(ROOTDDEP) $(ROOTD)

distclean::     distclean-rootd

##### extra rules ######
$(ROOTDDIRS)/rootd.o: $(ROOTDDIRS)/rootd.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(AUTHFLAGS) -o $@ -c $<
