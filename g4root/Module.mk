# Module.mk for g4root module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Andrei Gheata, 08/08/2006

MODDIR       := g4root
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

G4ROOTDIR      := $(MODDIR)
G4ROOTDIRS     := $(G4ROOTDIR)/src
G4ROOTDIRI     := $(G4ROOTDIR)/inc

##### libG4root #####
G4ROOTL1       := $(MODDIRI)/LinkDef.h
G4ROOTDS1      := $(MODDIRS)/G__G4root.cxx
G4ROOTDO1      := $(G4ROOTDS1:.cxx=.o)
G4ROOTDS       := $(G4ROOTDS1)
G4ROOTDO       := $(G4ROOTDO1)
G4ROOTDH       := $(G4ROOTDS:.cxx=.h)

G4ROOTH1       := TG4RootNavigator.h TG4RootSolid.h TG4RootDetectorConstruction.h
G4ROOTH2       := TG4RootNavMgr.h
G4ROOTH1       := $(patsubst %,$(MODDIRI)/%,$(G4ROOTH1))
G4ROOTH2       := $(patsubst %,$(MODDIRI)/%,$(G4ROOTH2))
G4ROOTH        := $(G4ROOTH1) $(G4ROOTH2)
G4ROOTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
G4ROOTO        := $(G4ROOTS:.cxx=.o)

G4ROOTDEP      := $(G4ROOTO:.o=.d) $(G4ROOTDO:.o=.d)

G4ROOTLIB      := $(LPATH)/libG4root.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(G4ROOTH))
ALLLIBS     += $(G4ROOTLIB)

# include all dependency files
INCLUDEFILES += $(G4ROOTDEP)

##### local rules #####
include/%.h:    $(G4ROOTDIRI)/%.h
		cp $< $@

$(G4ROOTLIB):     $(G4ROOTO) $(G4ROOTDO) $(ORDER_) $(MAINLIBS) $(G4ROOTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libG4geom.$(SOEXT) $@ "$(G4ROOTO) $(G4ROOTDO)" \
		   "$(G4ROOTLIBEXTRA)"

$(G4ROOTDS1):     $(G4ROOTH1) $(G4ROOTL1) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CXXFLAGS) -I$(G4ROOTINCDIR) $(G4ROOTH2) $(G4ROOTL1)

all-g4root:       $(G4ROOTLIB)

map-g4root:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(G4ROOTLIB) \
		   -d $(G4ROOTLIBDEP) -c $(G4ROOTLIB)

map::           

clean-g4root:
		@rm -f $(G4ROOTO) $(G4ROOTDO)

clean::         clean-g4root

distclean-g4root: clean-g4root
		@rm -f $(G4ROOTDEP) $(G4ROOTDS) $(G4ROOTDH) $(G4ROOTLIB)

distclean::     distclean-g4root

##### extra rules ######
$(G4ROOTO): CXXFLAGS += $(G4INCDIR:%=-I%) $(CLHEPINCDIR:%=-I%)
