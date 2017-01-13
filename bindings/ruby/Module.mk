# Module.mk for rubyroot module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Elias Athanasopoulos, 31/5/2004

MODNAME        := ruby
MODDIR         := $(ROOT_SRCDIR)/bindings/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

RUBYROOTDIR    := $(MODDIR)
RUBYROOTDIRS   := $(RUBYROOTDIR)/src
RUBYROOTDIRI   := $(RUBYROOTDIR)/inc

##### libRuby #####
RUBYROOTL      := $(MODDIRI)/LinkDef.h
RUBYROOTDS     := $(call stripsrc,$(MODDIRS)/G__Ruby.cxx)
RUBYROOTDO     := $(RUBYROOTDS:.cxx=.o)
RUBYROOTDH     := $(RUBYROOTDS:.cxx=.h)

RUBYROOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RUBYROOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RUBYROOTO      := $(call stripsrc,$(RUBYROOTS:.cxx=.o))

RUBYROOTDEP    := $(RUBYROOTO:.o=.d) $(RUBYROOTDO:.o=.d)

RUBYROOTLIB    := $(LPATH)/libRuby.$(SOEXT)
RUBYROOTMAP    := $(RUBYROOTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
RUBYROOTH_REL  := $(patsubst $(MODDIRI)/%.h,include/%.h,$(RUBYROOTH))
ALLHDRS        += $(RUBYROOTH_REL)
ALLLIBS        += $(RUBYROOTLIB)
ALLMAPS        += $(RUBYROOTMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(RUBYROOTH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module $(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(RUBYROOTLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif


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

$(call pcmrule,RUBYROOT)
	$(noop)

$(RUBYROOTDS):  $(RUBYROOTH) $(RUBYROOTL) $(ROOTCLINGEXE) $(call pcmdep,RUBYROOT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,RUBYROOT) -c $(RUBYROOTH) $(RUBYROOTL)

$(RUBYROOTMAP): $(RUBYROOTH) $(RUBYROOTL) $(ROOTCLINGEXE) $(call pcmdep,RUBYROOT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(RUBYROOTDS) $(call dictModule,RUBYROOT) -c $(RUBYROOTH) $(RUBYROOTL)

all-$(MODNAME): $(RUBYROOTLIB)

clean-$(MODNAME):
		@rm -f $(RUBYROOTO) $(RUBYROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RUBYROOTDEP) $(RUBYROOTDS) $(RUBYROOTDH) \
		   $(RUBYROOTLIB) $(RUBYROOTMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(RUBYROOTO): CXXFLAGS += $(RUBYINCDIR:%=-I%)
