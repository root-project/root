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

$(call pcmrule,RINT)
	$(noop)

$(RINTDS):      $(RINTH) $(RINTL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,RINT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE1) -f $@ $(call dictModule,RINT) -c -writeEmptyRootPCM $(RINTH) $(RINTL) && touch lib/libRint_rdict.pcm

$(RINTMAP):     $(RINTH) $(RINTL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,RINT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE1) -r $(RINTDS) $(call dictModule,RINT) -c $(RINTH) $(RINTL)

all-$(MODNAME): $(RINTLIB)

clean-$(MODNAME):
		@rm -f $(RINTO) $(RINTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RINTDEP) $(RINTDS) $(RINTDH) $(RINTLIB) $(RINTMAP)

distclean::     distclean-$(MODNAME)

ifeq ($(BUILDEDITLINE),yes)
$(RINTO): CXXFLAGS += -DR__BUILDEDITLINE
endif
