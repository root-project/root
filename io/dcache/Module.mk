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
DCACHEH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(DCACHEH))
ALLHDRS     += $(DCACHEH_REL)
ALLLIBS     += $(DCACHELIB)
ALLMAPS     += $(DCACHEMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(DCACHEH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Io_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(DCACHELIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

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

$(call pcmrule,DCACHE)
	$(noop)

$(DCACHEDS):    $(DCACHEH) $(DCACHEL) $(ROOTCLINGEXE) $(call pcmdep,DCACHE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,DCACHE) -c $(DCACHEH) $(DCACHEL)

$(DCACHEMAP):   $(DCACHEH) $(DCACHEL) $(ROOTCLINGEXE) $(call pcmdep,DCACHE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(DCACHEDS) $(call dictModule,DCACHE) -c $(DCACHEH) $(DCACHEL)

all-$(MODNAME): $(DCACHELIB)

clean-$(MODNAME):
	@rm -f $(DCACHEO) $(DCACHEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(DCACHEDEP) $(DCACHEDS) $(DCACHEDH) $(DCACHELIB) $(DCACHEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(DCACHEO) $(DCACHEDO): CXXFLAGS := $(filter-out -Wall,$(CXXFLAGS)) $(DCAPINCDIR:%=-I%)
ifneq ($(CXX:g++=),$(CXX))
   $(DCACHEO) $(DCACHEDO):  CXXFLAGS += -Wno-ignored-qualifiers
endif

