# Module.mk for rint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := rint
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RINTDIR      := $(MODDIR)
RINTDIRS     := $(RINTDIR)/src
RINTDIRI     := $(RINTDIR)/inc

##### libRint #####
RINTL        := $(MODDIRI)/LinkDef.h
RINTDS       := $(call stripsrc,$(MODDIRS)/G__Rint.cxx)
RINTDO       := $(RINTDS:.cxx=.o)
RINTDH       := $(RINTDS:.cxx=.h)

RINTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RINTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RINTO        := $(call stripsrc,$(RINTS:.cxx=.o))

RINTDEP      := $(RINTO:.o=.d) $(RINTDO:.o=.d)

RINTLIB      := $(LPATH)/libRint.$(SOEXT)
RINTMAP      := $(RINTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RINTH))
ALLLIBS     += $(RINTLIB)
ALLMAPS     += $(RINTMAP)

# include all dependency files
INCLUDEFILES += $(RINTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(RINTDIRI)/%.h
		cp $< $@

$(RINTLIB):     $(RINTO) $(RINTDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRint.$(SOEXT) $@ "$(RINTO) $(RINTDO)" \
		   "$(RINTLIBEXTRA)"

$(RINTDS):      $(RINTH) $(RINTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RINTH) $(RINTL)

$(RINTMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(RINTL)
		$(RLIBMAP) -o $@ -l $(RINTLIB) \
		   -d $(RINTLIBDEPM) -c $(RINTL)

all-$(MODNAME): $(RINTLIB) $(RINTMAP)

clean-$(MODNAME):
		@rm -f $(RINTO) $(RINTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RINTDEP) $(RINTDS) $(RINTDH) $(RINTLIB) $(RINTMAP)

distclean::     distclean-$(MODNAME)

ifeq ($(BUILDEDITLINE),yes)
$(RINTO): CXXFLAGS += -DR__BUILDEDITLINE
endif
