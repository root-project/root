# Module.mk for pyroot module
#
# Authors: Pere Mato, Wim Lavrijsen, 22/4/2004

MODDIR       := ruby
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RUBYROOTDIR    := $(MODDIR)
RUBYROOTDIRS   := $(RUBYROOTDIR)/src
RUBYROOTDIRI   := $(RUBYROOTDIR)/inc

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
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RUBYROOTH))
ALLLIBS     += $(RUBYROOTLIB)

# include all dependency files
INCLUDEFILES += $(RUBYROOTDEP)

##### local rules #####
include/%.h:    $(RUBYROOTDIRI)/%.h
		cp $< $@

#$(ROOTPY):      $(ROOTPYS)
#		cp $< $@

$(RUBYROOTLIB):   $(RUBYROOTO) $(RUBYROOTDO) $(MAINLIBS) $(ROOTPY)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		"$(SOFLAGS)" libRuby.$(SOEXT) $@ \
		"$(RUBYROOTO) $(RUBYROOTDO)" "$(RUBYLIBDIR) $(RUBYLIB)" \
                "$(RUBYLIBFLAGS)"

$(RUBYROOTDS):    $(RUBYROOTH) $(RUBYROOTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RUBYROOTH) $(RUBYROOTL)

$(RUBYROOTDO):    $(RUBYROOTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-ruby:     $(RUBYROOTLIB)

clean-ruby:
		@rm -f $(RUBYROOTO) $(RUBYROOTDO)

clean::         clean-ruby

distclean-ruby: clean-ruby
		@rm -f $(RUBYROOTDEP) $(RUBYROOTDS) $(RUBYROOTDH) $(RUBYROOTLIB)

distclean::     distclean-ruby

##### extra rules ######
$(RUBYROOTO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(RUBYINCDIR) -o $@ -c $<
