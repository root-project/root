# Module.mk for rubyroot module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Elias Athanasopoulos, 31/5/2004

MODDIR         := ruby
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
include/%.h:    $(RUBYROOTDIRI)/%.h
		cp $< $@

$(RUBYROOTLIB): $(RUBYROOTO) $(RUBYROOTDO) $(ORDER_) $(MAINLIBS) $(RUBYLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRuby.$(SOEXT) $@ \
		   "$(RUBYROOTO) $(RUBYROOTDO)" \
		   "$(RUBYLIBDIR) $(RUBYLIB) $(RUBYLIBEXTRA) $(CRYPTLIBS)"

$(RUBYROOTDS):  $(RUBYROOTH) $(RUBYROOTL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RUBYROOTH) $(RUBYROOTL)

$(RUBYROOTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(RUBYROOTL)
		$(RLIBMAP) -o $(RUBYROOTMAP) -l $(RUBYROOTLIB) \
		   -d $(RUBYROOTLIBDEPM) -c $(RUBYROOTL)

all-ruby:       $(RUBYROOTLIB) $(RUBYROOTMAP)

clean-ruby:
		@rm -f $(RUBYROOTO) $(RUBYROOTDO)

clean::         clean-ruby

distclean-ruby: clean-ruby
		@rm -f $(RUBYROOTDEP) $(RUBYROOTDS) $(RUBYROOTDH) \
		   $(RUBYROOTLIB) $(RUBYROOTMAP)

distclean::     distclean-ruby

##### extra rules ######
$(RUBYROOTO): CXXFLAGS += $(RUBYINCDIR:%=-I%)
