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

# used in the main Makefile
ALLHDRS        += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RUBYROOTH))
ALLLIBS        += $(RUBYROOTLIB)

# include all dependency files
INCLUDEFILES   += $(RUBYROOTDEP)

##### local rules #####
include/%.h:    $(RUBYROOTDIRI)/%.h
		cp $< $@

$(RUBYROOTLIB): $(RUBYROOTO) $(RUBYROOTDO) $(MAINLIBS) $(RUBYLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRuby.$(SOEXT) $@ \
		   "$(RUBYROOTO) $(RUBYROOTDO)" \
		   "$(RUBYLIBDIR) $(RUBYLIB) $(RUBYLIBEXTRA) $(CRYPTLIBS)"

$(RUBYROOTDS):  $(RUBYROOTH) $(RUBYROOTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RUBYROOTH) $(RUBYROOTL)

$(RUBYROOTDO):  $(RUBYROOTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-ruby:       $(RUBYROOTLIB)

map-ruby:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(RUBYROOTLIB) \
		   -d $(RUBYROOTLIBDEP) -c $(RUBYROOTL)

map::           map-ruby

clean-ruby:
		@rm -f $(RUBYROOTO) $(RUBYROOTDO)

clean::         clean-ruby

distclean-ruby: clean-ruby
		@rm -f $(RUBYROOTDEP) $(RUBYROOTDS) $(RUBYROOTDH) $(RUBYROOTLIB)

distclean::     distclean-ruby

##### extra rules ######
$(RUBYROOTO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(RUBYINCDIR:%=-I%) -o $@ -c $<
