# Module.mk for html module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := html
MODDIR       := $(ROOT_SRCDIR)/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HTMLDIR      := $(MODDIR)
HTMLDIRS     := $(HTMLDIR)/src
HTMLDIRI     := $(HTMLDIR)/inc

##### libHtml #####
HTMLL        := $(MODDIRI)/LinkDef.h
HTMLDS       := $(call stripsrc,$(MODDIRS)/G__Html.cxx)
HTMLDO       := $(HTMLDS:.cxx=.o)
HTMLDH       := $(HTMLDS:.cxx=.h)

HTMLH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HTMLS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HTMLO        := $(call stripsrc,$(HTMLS:.cxx=.o))

HTMLDEP      := $(HTMLO:.o=.d) $(HTMLDO:.o=.d)

HTMLLIB      := $(LPATH)/libHtml.$(SOEXT)
HTMLMAP      := $(HTMLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
HTMLH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(HTMLH))
ALLHDRS     += $(HTMLH_REL)
ALLLIBS     += $(HTMLLIB)
ALLMAPS     += $(HTMLMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(HTMLH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Html_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(HTMLLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(HTMLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HTMLDIRI)/%.h
		cp $< $@

$(HTMLLIB):     $(HTMLO) $(HTMLDO) $(ORDER_) $(MAINLIBS) $(HTMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHtml.$(SOEXT) $@ "$(HTMLO) $(HTMLDO)" \
		   "$(HTMLLIBEXTRA)"

$(call pcmrule,HTML)
	$(noop)

$(HTMLDS):      $(HTMLH_REL) $(HTMLL) $(ROOTCLINGEXE) $(call pcmdep,HTML)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,HTML) -c $(HTMLH) $(HTMLL)

$(HTMLMAP):     $(HTMLH_REL) $(HTMLL) $(ROOTCLINGEXE) $(call pcmdep,HTML)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HTMLDS) $(call dictModule,HTML) -c $(HTMLH) $(HTMLL)

all-$(MODNAME): $(HTMLLIB)

clean-$(MODNAME):
		@rm -f $(HTMLO) $(HTMLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HTMLDEP) $(HTMLDS) $(HTMLDH) $(HTMLLIB) $(HTMLMAP)

distclean::     distclean-$(MODNAME)
