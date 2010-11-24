# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: David Gonzalez Maline, 21/4/2008

MODNAME      := genetic
MODDIR       := math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GENETICDIR  := $(MODDIR)
GENETICDIRS := $(GENETICDIR)/src
GENETICDIRI := $(GENETICDIR)/inc/Math

##### libGenetic #####
GENETICL    := $(MODDIRI)/LinkDef.h
#GENETICLINC :=
GENETICDS   := $(MODDIRS)/G__Genetic.cxx
GENETICDO   := $(GENETICDS:.cxx=.o)
GENETICDH   := $(GENETICDS:.cxx=.h)

GENETICDH1  :=  $(MODDIRI)/Math/GeneticMinimizer.h \

GENETICH   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))
GENETICS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GENETICO    := $(GENETICS:.cxx=.o)

GENETICDEP  := $(GENETICO:.o=.d)  $(GENETICDO:.o=.d)

GENETICLIB  := $(LPATH)/libGenetic.$(SOEXT)
GENETICMAP  := $(GENETICLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(GENETICH))
ALLLIBS      += $(GENETICLIB)
ALLMAPS      += $(GENETICMAP)

# include all dependency files
INCLUDEFILES += $(GENETICDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME) check-$(MODNAME)

include/Math/%.h: $(GENETICDIRI)/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

$(GENETICLIB): $(GENETICO) $(GENETICDO) $(ORDER_) $(MAINLIBS) $(GENETICLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libGenetic.$(SOEXT) $@     \
		   "$(GENETICO) $(GENETICDO)" \
		   "$(GENETICLIBEXTRA)"

$(GENETICDS):  $(GENETICDH1) $(GENETICL) $(GENETICLINC) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		@echo "for files $(GENETICDH1)"
		$(ROOTCINTTMP) -f $@ -c $(GENETICDH1) $(GENETICL)

$(GENETICMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(GENETICL) $(GENETICLINC)
		$(RLIBMAP) -o $(GENETICMAP) -l $(GENETICLIB) \
		   -d $(GENETICLIBDEPM) -c $(GENETICL) $(GENETICLINC)

ifneq ($(ICC_MAJOR),)
# silence warning messages about subscripts being out of range
$(GENETICDO):   CXXFLAGS += -wd175 -I$(GENETICDIRI)
else
$(GENETICDO):   CXXFLAGS += -I$(GENETICDIRI)
endif

all-$(MODNAME): $(GENETICLIB) $(GENETICMAP)

clean-$(MODNAME):
		@rm -f $(GENETICO) $(GENETICDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GENETICDEP) $(GENETICDS) $(GENETICDH) \
		   $(GENETICLIB) $(GENETICMAP)
		@rm -rf include/Math
		-@cd $(GENETICDIR)/test && $(MAKE) distclean

distclean::     distclean-$(MODNAME)

test-$(MODNAME): all-$(MODNAME)
		@cd $(GENETICDIR)/test && $(MAKE)

check-$(MODNAME): test-$(MODNAME)
		@cd $(GENETICDIR)/test && $(MAKE)
