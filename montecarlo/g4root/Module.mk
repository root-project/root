# Module.mk for g4root module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Andrei Gheata, 08/08/2006

MODNAME      := g4root
MODDIR       := montecarlo/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ifdef G4ROOT_DEBUG
  CXXFLAGS   += -DG4ROOT_DEBUG
endif
   
G4ROOTDIR    := $(MODDIR)
G4ROOTDIRS   := $(G4ROOTDIR)/src
G4ROOTDIRI   := $(G4ROOTDIR)/inc

##### libG4root #####
G4ROOTL1     := $(MODDIRI)/LinkDef.h
G4ROOTDS1    := $(MODDIRS)/G__G4root.cxx
G4ROOTDO1    := $(G4ROOTDS1:.cxx=.o)
G4ROOTDS     := $(G4ROOTDS1)
G4ROOTDO     := $(G4ROOTDO1)
G4ROOTDH     := $(G4ROOTDS:.cxx=.h)

G4ROOTH1     := TG4RootNavigator.h TG4RootSolid.h TG4RootDetectorConstruction.h
G4ROOTH2     := TG4RootNavMgr.h
G4ROOTH1     := $(patsubst %,$(MODDIRI)/%,$(G4ROOTH1))
G4ROOTH2     := $(patsubst %,$(MODDIRI)/%,$(G4ROOTH2))
G4ROOTH      := $(G4ROOTH1) $(G4ROOTH2)
G4ROOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
G4ROOTO      := $(G4ROOTS:.cxx=.o)

G4ROOTDEP    := $(G4ROOTO:.o=.d) $(G4ROOTDO:.o=.d)

G4ROOTLIB    := $(LPATH)/libG4root.$(SOEXT)
G4ROOTMAP    := $(G4ROOTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(G4ROOTH))
ALLLIBS     += $(G4ROOTLIB)
ALLMAPS     += $(G4ROOTMAP)

# include all dependency files
INCLUDEFILES += $(G4ROOTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(G4ROOTDIRI)/%.h
		cp $< $@

$(G4ROOTLIB):   $(G4ROOTO) $(G4ROOTDO) $(ORDER_) $(MAINLIBS) $(G4ROOTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libG4root.$(SOEXT) $@ "$(G4ROOTO) $(G4ROOTDO)" \
		   "$(G4ROOTLIBEXTRA)"

$(G4ROOTDS1):   $(G4ROOTH1) $(G4ROOTL1) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CXXFLAGS) -I$(G4ROOTINCDIR) \
		   $(G4ROOTH2) $(G4ROOTL1)

$(G4ROOTMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(G4ROOTL)
		$(RLIBMAP) -o $(G4ROOTMAP) -l $(G4ROOTLIB) \
		   -d $(G4ROOTLIBDEPM) -c $(G4ROOTL)

all-$(MODNAME): $(G4ROOTLIB) $(G4ROOTMAP)

clean-$(MODNAME):
		@rm -f $(G4ROOTO) $(G4ROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(G4ROOTDEP) $(G4ROOTDS) $(G4ROOTDH) $(G4ROOTLIB) $(G4ROOTMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(G4ROOTO): CXXFLAGS += $(G4INCDIR:%=-I%) $(CLHEPINCDIR:%=-I%)
