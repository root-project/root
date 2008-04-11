# Module.mk for graf module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := graf
MODDIR       := graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GRAFDIR      := $(MODDIR)
GRAFDIRS     := $(GRAFDIR)/src
GRAFDIRI     := $(GRAFDIR)/inc

##### libGraf #####
GRAFL1       := $(MODDIRI)/LinkDef1.h
GRAFL2       := $(MODDIRI)/LinkDef2.h
GRAFDS1      := $(MODDIRS)/G__Graf1.cxx
GRAFDS2      := $(MODDIRS)/G__Graf2.cxx
GRAFDO1      := $(GRAFDS1:.cxx=.o)
GRAFDO2      := $(GRAFDS2:.cxx=.o)
GRAFDS       := $(GRAFDS1) $(GRAFDS2)
GRAFDO       := $(GRAFDO1) $(GRAFDO2)
GRAFDH       := $(GRAFDS:.cxx=.h)

GRAFH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GRAFHD       := $(filter-out $(MODDIRI)/TTF.h,$(GRAFH))
GRAFHD       := $(filter-out $(MODDIRI)/TText.h,$(GRAFHD))
GRAFHD       := $(filter-out $(MODDIRI)/TLatex.h,$(GRAFHD))
GRAFS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GRAFO        := $(GRAFS:.cxx=.o)

GRAFDEP      := $(GRAFO:.o=.d) $(GRAFDO:.o=.d)

GRAFLIB      := $(LPATH)/libGraf.$(SOEXT)
GRAFMAP      := $(GRAFLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GRAFH))
ALLLIBS     += $(GRAFLIB)
ALLMAPS     += $(GRAFMAP)

# include all dependency files
INCLUDEFILES += $(GRAFDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GRAFDIRI)/%.h
		cp $< $@

$(GRAFLIB):     $(GRAFO) $(GRAFDO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) $(GRAFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGraf.$(SOEXT) $@ \
		   "$(GRAFO) $(GRAFDO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) $(GRAFLIBEXTRA)"

$(GRAFDS1):     $(GRAFHD) $(GRAFL1) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GRAFHD) $(GRAFL1)
$(GRAFDS2):     $(GRAFH) $(GRAFL2) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FREETYPEINC) $(GRAFH) $(GRAFL2)

$(GRAFMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(GRAFL1) $(GRAFL2)
		$(RLIBMAP) -o $(GRAFMAP) -l $(GRAFLIB) \
		   -d $(GRAFLIBDEPM) -c $(GRAFL1) $(GRAFL2)

all-$(MODNAME): $(GRAFLIB) $(GRAFMAP)

clean-$(MODNAME):
		@rm -f $(GRAFO) $(GRAFDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GRAFDEP) $(GRAFDS) $(GRAFDH) $(GRAFLIB) $(GRAFMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GRAFDO2):     $(FREETYPEDEP)
$(GRAFDO2):     OPT = $(NOOPT)
$(GRAFDO2):     CXXFLAGS += $(FREETYPEINC)

$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o: \
                $(FREETYPEDEP)
$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o: \
                CXXFLAGS += $(FREETYPEINC)

ifeq ($(PLATFORM),win32)
ifeq (,$(findstring $(VC_MAJOR),14 15))
$(GRAFDIRS)/TLatex.o: OPT = $(NOOPT)
endif
endif
