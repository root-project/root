# Module.mk for utils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := utils
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UTILSDIR     := $(MODDIR)
UTILSDIRS    := $(UTILSDIR)/src
UTILSDIRI    := $(UTILSDIR)/inc

##### rootcint #####
ROOTCINTS    := $(MODDIRS)/rootcint.cxx
ROOTCINTO    := $(ROOTCINTS:.cxx=.o)
ROOTCINTDEP  := $(ROOTCINTO:.o=.d)
ROOTCINTTMPO := $(ROOTCINTS:.cxx=_tmp.o)
ROOTCINTTMP  := $(MODDIRS)/rootcint_tmp$(EXEEXT)
ROOTCINT     := bin/rootcint$(EXEEXT)

# include all dependency files
INCLUDEFILES += $(ROOTCINTDEP)

##### local rules #####
$(ROOTCINT):    $(CINTLIB) $(ROOTCINTO) $(MAKEINFO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTCINTO) \
		   $(RPATH) $(CINTLIBS) $(CILIBS)

$(ROOTCINTTMP): $(CINTTMPO) $(ROOTCINTS) $(MAKEINFO)
		$(CXX) $(OPT) $(CXXFLAGS) -UHAVE_CONFIG -DROOTBUILD \
			-c $(ROOTCINTS) -o $(ROOTCINTTMPO)
		$(LD) $(LDFLAGS) -o $@ \
			$(ROOTCINTTMPO) $(CINTTMPO) $(CILIBS)

all-utils:      $(ROOTCINTTMP) $(ROOTCINT)

clean-utils:
		@rm -f $(ROOTCINTTMPO) $(ROOTCINTO)

clean::         clean-utils

distclean-utils: clean-utils
		@rm -f $(ROOTCINTDEP) $(ROOTCINTTMP) $(ROOTCINT)

distclean::     distclean-utils
