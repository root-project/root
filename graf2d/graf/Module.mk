# Module.mk for graf module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := graf
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GRAFDIR      := $(MODDIR)
GRAFDIRS     := $(GRAFDIR)/src
GRAFDIRI     := $(GRAFDIR)/inc

##### libGraf #####
GRAFL        := $(MODDIRI)/LinkDef.h
GRAFDS       := $(call stripsrc,$(MODDIRS)/G__Graf.cxx)
GRAFDO       := $(GRAFDS:.cxx=.o)
GRAFDH       := $(GRAFDS:.cxx=.h)

GRAFH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GRAFS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GRAFO        := $(call stripsrc,$(GRAFS:.cxx=.o))

GRAFDEP      := $(GRAFO:.o=.d) $(GRAFDO:.o=.d)

GRAFLIB      := $(LPATH)/libGraf.$(SOEXT)
GRAFMAP      := $(GRAFLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
GRAFH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(GRAFH))
ALLHDRS     += $(GRAFH_REL)
ALLLIBS     += $(GRAFLIB)
ALLMAPS     += $(GRAFMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(GRAFH_REL))
  # FIXME: TTF.h is a non-module header. It depends on preprocessor to figure out
  # whether it should FT_Vector_ and FT_BBox_.
  CXXMODULES_HEADERS := $(subst header \"TTF.h\",textual header \"TTF.h\",$(CXXMODULES_HEADERS))
  CXXMODULES_MODULEMAP_CONTENTS += module Grad2d_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(GRAFLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(GRAFDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GRAFDIRI)/%.h
		cp $< $@

$(GRAFLIB):     $(GRAFO) $(GRAFDO) $(MATHTEXTLIBDEP) $(FREETYPEDEP) $(ORDER_) \
                   $(MAINLIBS) $(GRAFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGraf.$(SOEXT) $@ \
		   "$(GRAFO) $(GRAFDO)" \
		   "$(GRAFLIBEXTRA) $(MATHTEXTLIB) $(FREETYPELDFLAGS) $(FREETYPELIB)"

$(call pcmrule,GRAF)
	$(noop)

$(GRAFDS):      $(GRAFH) $(GRAFL) $(ROOTCLINGEXE) $(call pcmdep,GRAF)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GRAF) -c -writeEmptyRootPCM $(CINTFLAGS) $(GRAFH) $(GRAFL)

$(GRAFMAP):     $(GRAFH) $(GRAFL) $(ROOTCLINGEXE) $(call pcmdep,GRAF)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GRAFDS) $(call dictModule,GRAF) -c $(CINTFLAGS) $(GRAFH) $(GRAFL)

all-$(MODNAME): $(GRAFLIB)

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
$(GRAFDS):     CINTFLAGS += $(FREETYPEINC)

$(call stripsrc,$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o $(GRAFDIRS)/TMathText.o): \
                $(FREETYPEDEP)
$(call stripsrc,$(GRAFDIRS)/TTF.o $(GRAFDIRS)/TText.o $(GRAFDIRS)/TLatex.o $(GRAFDIRS)/TMathText.o): \
                CXXFLAGS += $(FREETYPEINC)

ifeq ($(PLATFORM),win32)
ifeq (,$(findstring $(VC_MAJOR),14 15))
$(call stripsrc,$(GRAFDIRS)/TLatex.o): OPT = $(NOOPT)
endif
endif
