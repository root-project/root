# Module.mk for dcache module
#
# Author: Grzegorz Mazur <mazur@mail.desy.de>, 16/1/2002

MODDIR       := dcache
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

DCACHEDIR    := $(MODDIR)
DCACHEDIRS   := $(DCACHEDIR)/src
DCACHEDIRI   := $(DCACHEDIR)/inc

##### libDCache #####
DCACHEL      := $(MODDIRI)/LinkDef.h
DCACHEDS     := $(MODDIRS)/G__DCache.cxx
DCACHEDO     := $(DCACHEDS:.cxx=.o)
DCACHEDH     := $(DCACHEDS:.cxx=.h)

DCACHEH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
DCACHES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
DCACHEO      := $(DCACHES:.cxx=.o)

DCACHEDEP    := $(DCACHEO:.o=.d) $(DCACHEDO:.o=.d)

DCACHELIB    := $(LPATH)/libDCache.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(DCACHEH))
ALLLIBS     += $(DCACHELIB)

# include all dependency files
INCLUDEFILES += $(DCACHEDEP)

##### local rules #####
include/%.h:    $(DCACHEDIRI)/%.h
	cp $< $@

$(DCACHELIB):   $(DCACHEO) $(DCACHEDO) $(MAINLIBS)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
	"$(SOFLAGS)" libDCache.$(SOEXT) $@ "$(DCACHEO) $(DCACHEDO)" \
	"$(DCACHELIBEXTRA) $(DCAPLIBDIR) $(DCAPLIB)"

$(DCACHEDS):    $(DCACHEH) $(DCACHEL) $(ROOTCINTTMP)
	@echo "Generating dictionary $@..."
	$(ROOTCINTTMP) -f $@ -c $(DCACHEH) $(DCACHEL)

$(DCACHEDO):    $(DCACHEDS)
	$(CXX) $(NOOPT) $(CXXFLAGS) $(DCAPINCDIR:%=-I%) -I. -o $@ -c $<

all-dcache:     $(DCACHELIB)

map-dcache:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(DCACHELIB) \
		   -d $(DCACHELIBDEP) -c $(DCACHEL)

map::           map-dcache

clean-dcache:
	@rm -f $(DCACHEO) $(DCACHEDO)

clean::         clean-dcache

distclean-dcache: clean-dcache
	@rm -f $(DCACHEDEP) $(DCACHEDS) $(DCACHEDH) $(DCACHELIB)

distclean::     distclean-dcache

##### extra rules ######
$(DCACHEO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(DCAPINCDIR:%=-I%) -o $@ -c $<
