# Module.mk for roostats module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Kyle Cranmer

MODNAME      := roostats
MODDIR       := roofit/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOSTATSDIR  := $(MODDIR)
ROOSTATSDIRS := $(ROOSTATSDIR)/src
ROOSTATSDIRI := $(ROOSTATSDIR)/inc

##### libRooStats #####
ROOSTATSL    := $(MODDIRI)/LinkDef.h
ROOSTATSDS   := $(MODDIRS)/G__RooStats.cxx
ROOSTATSDO   := $(ROOSTATSDS:.cxx=.o)
ROOSTATSDH   := $(ROOSTATSDS:.cxx=.h)

ROOSTATSH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ROOSTATSS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOSTATSO    := $(ROOSTATSS:.cxx=.o)

ROOSTATSDEP  := $(ROOSTATSO:.o=.d) $(ROOSTATSDO:.o=.d)

ROOSTATSLIB  := $(LPATH)/libRooStats.$(SOEXT)
ROOSTATSMAP  := $(ROOSTATSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOSTATSH))
ALLLIBS      += $(ROOSTATSLIB)
ALLMAPS      += $(ROOSTATSMAP)

# include all dependency files
INCLUDEFILES += $(ROOSTATSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ROOSTATSDIRI)/%.h
		cp $< $@

$(ROOSTATSLIB): $(ROOSTATSO) $(ROOSTATSDO) $(ORDER_) $(MAINLIBS) \
                $(ROOSTATSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooStats.$(SOEXT) $@ \
		   "$(ROOSTATSO) $(ROOSTATSDO)" \
		   "$(ROOSTATSLIBEXTRA)"

$(ROOSTATSDS):  $(ROOSTATSH) $(ROOSTATSL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOSTATSH) $(ROOSTATSL)

$(ROOSTATSMAP): $(RLIBMAP) $(MAKEFILEDEP) $(ROOSTATSL)
		$(RLIBMAP) -o $(ROOSTATSMAP) -l $(ROOSTATSLIB) \
		   -d $(ROOSTATSLIBDEPM) -c $(ROOSTATSL)

all-$(MODNAME): $(ROOSTATSLIB) $(ROOSTATSMAP)

clean-$(MODNAME):
		@rm -f $(ROOSTATSO) $(ROOSTATSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(ROOSTATSDEP) $(ROOSTATSLIB) $(ROOSTATSMAP) \
		   $(ROOSTATSDS) $(ROOSTATSDH)

distclean::     distclean-$(MODNAME)
