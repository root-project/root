# Module.mk for dcache module
#
# Author: Grzegorz Mazur <mazur@mail.desy.de>, 16/1/2002

MODNAME      := dcache
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

DCACHEDIR    := $(MODDIR)
DCACHEDIRS   := $(DCACHEDIR)/src
DCACHEDIRI   := $(DCACHEDIR)/inc

##### libDCache #####
DCACHEL      := $(MODDIRI)/LinkDef.h
DCACHEDS     := $(call stripsrc,$(MODDIRS)/G__DCache.cxx)
DCACHEDO     := $(DCACHEDS:.cxx=.o)
DCACHEDH     := $(DCACHEDS:.cxx=.h)

DCACHEH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
DCACHES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
DCACHEO      := $(call stripsrc,$(DCACHES:.cxx=.o))

DCACHEDEP    := $(DCACHEO:.o=.d) $(DCACHEDO:.o=.d)

DCACHELIB    := $(LPATH)/libDCache.$(SOEXT)
DCACHEMAP    := $(DCACHELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(DCACHEH))
ALLLIBS     += $(DCACHELIB)
ALLMAPS     += $(DCACHEMAP)

# include all dependency files
INCLUDEFILES += $(DCACHEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(DCACHEDIRI)/%.h
		cp $< $@

$(DCACHELIB):   $(DCACHEO) $(DCACHEDO) $(ORDER_) $(MAINLIBS) $(DCACHELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libDCache.$(SOEXT) $@ "$(DCACHEO) $(DCACHEDO)" \
		   "$(DCACHELIBEXTRA) $(DCAPLIBDIR) $(DCAPLIB)"

$(DCACHEDS):    $(DCACHEH) $(DCACHEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(DCACHEH) $(DCACHEL)

$(DCACHEMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(DCACHEL)
		$(RLIBMAP) -o $@ -l $(DCACHELIB) \
		   -d $(DCACHELIBDEPM) -c $(DCACHEL)

all-$(MODNAME): $(DCACHELIB) $(DCACHEMAP)

clean-$(MODNAME):
	@rm -f $(DCACHEO) $(DCACHEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(DCACHEDEP) $(DCACHEDS) $(DCACHEDH) $(DCACHELIB) $(DCACHEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(DCACHEO) $(DCACHEDO): CXXFLAGS := $(filter-out -Wall,$(CXXFLAGS)) $(DCAPINCDIR:%=-I%)
ifeq ($(GCC_MAJOR),4)
ifeq ($(subst $(GCC_MINOR),,0 1 2),0 1 2)
# GCC >= 4.3
   $(DCACHEO) $(DCACHEDO):  CXXFLAGS += -Wno-ignored-qualifiers
endif
endif

