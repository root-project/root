# Module.mk for rubyroot module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Elias Athanasopoulos, 31/5/2004

MODNAME        := ruby
MODDIR         := bindings/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

RUBYROOTDIR    := $(MODDIR)
RUBYROOTDIRS   := $(RUBYROOTDIR)/src
RUBYROOTDIRI   := $(RUBYROOTDIR)/inc

##### libRuby #####
RUBYROOTL      := $(MODDIRI)/LinkDef.h
RUBYROOTDS     := $(MODDIRS)/G__Ruby.cxx
RUBYROOTDO     := $(RUBYROOTDS:.cxx=.o)
RUBYROOTDH     := $(RUBYROOTDS:.cxx=.h)

RUBYROOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RUBYROOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RUBYROOTO      := $(RUBYROOTS:.cxx=.o)

RUBYROOTDEP    := $(RUBYROOTO:.o=.d) $(RUBYROOTDO:.o=.d)

RUBYROOTLIB    := $(LPATH)/libRuby.$(SOEXT)
RUBYROOTMAP    := $(RUBYROOTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS        += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RUBYROOTH))
ALLLIBS        += $(RUBYROOTLIB)
ALLMAPS        += $(RUBYROOTMAP)

# include all dependency files
INCLUDEFILES   += $(RUBYROOTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(RUBYROOTDIRI)/%.h
		cp $< $@

$(RUBYROOTLIB): $(RUBYROOTO) $(RUBYROOTDO) $(ORDER_) $(MAINLIBS) $(RUBYLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRuby.$(SOEXT) $@ \
		   "$(RUBYROOTO) $(RUBYROOTDO)" \
		   "$(RUBYLIBDIR) $(RUBYLIB) $(RUBYLIBEXTRA) $(CRYPTLIBS)"

$(RUBYROOTDS):  $(RUBYROOTH) $(RUBYROOTL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RUBYROOTH) $(RUBYROOTL)

$(RUBYROOTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(RUBYROOTL)
		$(RLIBMAP) -o $(RUBYROOTMAP) -l $(RUBYROOTLIB) \
		   -d $(RUBYROOTLIBDEPM) -c $(RUBYROOTL)

all-$(MODNAME): $(RUBYROOTLIB) $(RUBYROOTMAP)

clean-$(MODNAME):
		@rm -f $(RUBYROOTO) $(RUBYROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RUBYROOTDEP) $(RUBYROOTDS) $(RUBYROOTDH) \
		   $(RUBYROOTLIB) $(RUBYROOTMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(RUBYROOTO): CXXFLAGS += $(RUBYINCDIR:%=-I%) -Iinclude/cint
