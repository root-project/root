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
GRAFL        := $(MODDIRI)/LinkDef.h
GRAFDS       := $(MODDIRS)/G__Graf.cxx
GRAFDO       := $(GRAFDS:.cxx=.o)
GRAFDH       := $(GRAFDS:.cxx=.h)

GRAFH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
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

$(GRAFDS):      $(GRAFH) $(GRAFL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GRAFH) $(GRAFL)

$(GRAFMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(GRAFL)
		$(RLIBMAP) -o $(GRAFMAP) -l $(GRAFLIB) \
		   -d $(GRAFLIBDEPM) -c $(GRAFL)

all-$(MODNAME): $(GRAFLIB) $(GRAFMAP)

clean-$(MODNAME):
		@rm -f $(GRAFO) $(GRAFDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GRAFDEP) $(GRAFDS) $(GRAFDH) $(GRAFLIB) $(GRAFMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GRAFDO):     $(FREETYPEDEP)
$(GRAFDO):     OPT = $(NOOPT)
$(GRAFDO):     CXXFLAGS += $(FREETYPEINC)

$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o: \
                $(FREETYPEDEP)
$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o: \
                CXXFLAGS += $(FREETYPEINC)

ifeq ($(PLATFORM),win32)
ifeq (,$(findstring $(VC_MAJOR),14 15))
$(GRAFDIRS)/TLatex.o: OPT = $(NOOPT)
endif
endif
