

# Module.mk for roostats module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Kyle Cranmer

MODNAME      := roostats
MODDIR       := $(ROOT_SRCDIR)/roofit/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOSTATSDIR  := $(MODDIR)
ROOSTATSDIRS := $(ROOSTATSDIR)/src
ROOSTATSDIRI := $(ROOSTATSDIR)/inc

##### libRooStats #####
ROOSTATSL    := $(MODDIRI)/LinkDef.h
ROOSTATSDS   := $(call stripsrc,$(MODDIRS)/G__RooStats.cxx)
ROOSTATSDO   := $(ROOSTATSDS:.cxx=.o)
ROOSTATSDH   := $(ROOSTATSDS:.cxx=.h)

ROOSTATSH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/RooStats/*.h))
ROOSTATSS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOSTATSO    := $(call stripsrc,$(ROOSTATSS:.cxx=.o))

ROOSTATSDEP  := $(ROOSTATSO:.o=.d) $(ROOSTATSDO:.o=.d)

ROOSTATSLIB  := $(LPATH)/libRooStats.$(SOEXT)
ROOSTATSMAP  := $(ROOSTATSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/RooStats/%.h,include/RooStats/%.h,$(ROOSTATSH))
ALLLIBS      += $(ROOSTATSLIB)
ALLMAPS      += $(ROOSTATSMAP)

# include all dependency files
INCLUDEFILES += $(ROOSTATSDEP)

#needed since include are in inc and not inc/RooStats
#ROOSTATSH_DIC   := $(subst $(MODDIRI),include/RooStats,$(ROOSTATSH))

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/RooStats/%.h: $(ROOSTATSDIRI)/RooStats/%.h
		@(if [ ! -d "include/RooStats" ]; then    \
		   mkdir -p include/RooStats;             \
		fi)
		cp $< $@

$(ROOSTATSLIB): $(ROOSTATSO) $(ROOSTATSDO) $(ORDER_) $(MAINLIBS) \
                $(ROOSTATSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooStats.$(SOEXT) $@ \
		   "$(ROOSTATSO) $(ROOSTATSDO)" \
		   "$(ROOSTATSLIBEXTRA)"

$(ROOSTATSDS):  $(ROOSTATSH) $(ROOSTATSL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOSTATSH) $(ROOSTATSL)

$(ROOSTATSMAP): $(RLIBMAP) $(MAKEFILEDEP) $(ROOSTATSL)
		$(RLIBMAP) -o $@ -l $(ROOSTATSLIB) \
		   -d $(ROOSTATSLIBDEPM) -c $(ROOSTATSL)

all-$(MODNAME): $(ROOSTATSLIB) $(ROOSTATSMAP)

clean-$(MODNAME):
		@rm -f $(ROOSTATSO) $(ROOSTATSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOSTATSDEP) $(ROOSTATSLIB) $(ROOSTATSMAP) \
		   $(ROOSTATSDS) $(ROOSTATSDH)
		@rm -rf include/RooStats

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(ROOSTATSDO): NOOPT = $(OPT)
