# Module.mk for peac module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Maarten Ballintijn 18/10/2004

MODDIR       := peac
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PEACDIR      := $(MODDIR)
PEACDIRS     := $(PEACDIR)/src
PEACDIRI     := $(PEACDIR)/inc

##### libPeac #####
PEACL        := $(MODDIRI)/LinkDef.h
PEACDS       := $(MODDIRS)/G__Peac.cxx
PEACDO       := $(PEACDS:.cxx=.o)
PEACDH       := $(PEACDS:.cxx=.h)

PEACH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PEACS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PEACO        := $(PEACS:.cxx=.o)

PEACDEP      := $(PEACO:.o=.d) $(PEACDO:.o=.d)

PEACLIB      := $(LPATH)/libPeac.$(SOEXT)

##### libPeacGui #####
PEACGUIL     := $(MODDIRI)/LinkDefGui.h
PEACGUIDS    := $(MODDIRS)/G__PeacGui.cxx
PEACGUIDO    := $(PEACGUIDS:.cxx=.o)
PEACGUIDH    := $(PEACGUIDS:.cxx=.h)

PEACGUIH     := $(MODDIRI)/TProofStartupDialog.h
PEACGUIS     := $(MODDIRS)/TProofStartupDialog.cxx
PEACGUIO     := $(PEACGUIS:.cxx=.o)

PEACGUIDEP   := $(PEACGUIO:.o=.d) $(PEACGUIDO:.o=.d)

PEACGUILIB   := $(LPATH)/libPeacGui.$(SOEXT)

# remove GUI files from PEAC files
PEACH        := $(filter-out $(PEACGUIH),$(PEACH))
PEACS        := $(filter-out $(PEACGUIS),$(PEACS))
PEACO        := $(filter-out $(PEACGUIO),$(PEACO))
PEACDEP      := $(filter-out $(PEACGUIDEP),$(PEACDEP))

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PEACH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PEACGUIH))
ALLLIBS     += $(PEACLIB) $(PEACGUILIB)

# include all dependency files
INCLUDEFILES += $(PEACDEP) $(PEACGUIDEP)

##### local rules #####
include/%.h:    $(PEACDIRI)/%.h
		cp $< $@

$(PEACLIB):     $(PEACO) $(PEACDO) $(MAINLIBS) $(PEACLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPeac.$(SOEXT) $@ "$(PEACO) $(PEACDO)" \
		   "$(PEACLIBEXTRA)"

$(PEACDS):      $(PEACH) $(PEACL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PEACH) $(PEACL)

$(PEACDO):      $(PEACDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

$(PEACGUILIB):  $(PEACGUIO) $(PEACGUIDO) $(MAINLIBS) $(PEACGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPeacGui.$(SOEXT) $@ \
		   "$(PEACGUIO) $(PEACGUIDO)" \
		   "$(PEACGUILIBEXTRA)"

$(PEACGUIDS):   $(PEACGUIH) $(PEACGUIL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PEACGUIH) $(PEACGUIL)

$(PEACGUIDO):   $(PEACGUIDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-peac:       $(PEACLIB) $(PEACGUILIB)

map-peac:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PEACLIB) \
		   -d $(PEACLIBDEP) -c $(PEACL)

map-peacgui:    $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PEACGUILIB) \
		   -d $(PEACGUILIBDEP) -c $(PEACGUIL)

map::           map-peac map-peacgui

clean-peac:
		@rm -f $(PEACO) $(PEACDO) $(PEACGUIO) $(PEACGUIDO)

clean::         clean-peac

distclean-peac: clean-peac
		@rm -f $(PEACDEP) $(PEACDS) $(PEACDH) $(PEACLIB) \
		   $(PEACGUIDEP) $(PEACGUIDS) $(PEACGUIDH) $(PEACGUILIB)

distclean::     distclean-peac

##### extra rules ######
$(PEACO):       %.o: %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(CLARENSINC) -o $@ -c $<
