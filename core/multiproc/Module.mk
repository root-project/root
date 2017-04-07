# Module.mk for multiproc module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := multiproc
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MULTIPROCDIR    := $(MODDIR)
MULTIPROCDIRS   := $(MULTIPROCDIR)/src
MULTIPROCDIRI   := $(MULTIPROCDIR)/inc

##### libMultiProc #####
MULTIPROCL      := $(MODDIRI)/LinkDef.h
MULTIPROCDS     := $(call stripsrc,$(MODDIRS)/G__MultiProc.cxx)
MULTIPROCDO     := $(MULTIPROCDS:.cxx=.o)
MULTIPROCDH     := $(MULTIPROCDS:.cxx=.h)

MULTIPROCH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h $(MODDIRI)/ROOT/*.hxx))
MULTIPROCS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MULTIPROCO      := $(call stripsrc,$(MULTIPROCS:.cxx=.o))

MULTIPROCDEP    := $(MULTIPROCO:.o=.d) $(MULTIPROCDO:.o=.d)

MULTIPROCLIB    := $(LPATH)/libMultiProc.$(SOEXT)
MULTIPROCMAP    := $(MULTIPROCLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%,include/%,$(MULTIPROCH) $(MULTIPROCH_EXT))
ALLLIBS      += $(MULTIPROCLIB)
ALLMAPS      += $(MULTIPROCMAP)

CXXFLAGS     += $(OSMULTIPROCFLAG)
CFLAGS       += $(OSMULTIPROCFLAG)

# include all dependency files
INCLUDEFILES += $(MULTIPROCDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MULTIPROCDIRI)/%.h
		cp $< $@

include/%.hxx:  $(MULTIPROCDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@

$(MULTIPROCLIB):   $(MULTIPROCO) $(MULTIPROCDO) $(ORDER_) $(MAINLIBS) $(MULTIPROCLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMultiProc.$(SOEXT) $@ "$(MULTIPROCO) $(MULTIPROCDO)" \
		   "$(MULTIPROCLIBEXTRA) $(OSMULTIPROCLIBDIR) $(OSMULTIPROCLIB)"

$(call pcmrule,MULTIPROC)
	$(noop)

$(MULTIPROCDS):    $(MULTIPROCH) $(MULTIPROCL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,MULTIPROC)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,MULTIPROC) -c $(MULTIPROCH) $(MULTIPROCL)

$(MULTIPROCMAP):   $(MULTIPROCH) $(MULTIPROCL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,MULTIPROC)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE1) -r $(MULTIPROCDS) $(call dictModule,MULTIPROC) -c $(MULTIPROCH) $(MULTIPROCL)

all-$(MODNAME): $(MULTIPROCLIB)

clean-$(MODNAME):
		@rm -f $(MULTIPROCO) $(MULTIPROCDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MULTIPROCDEP) $(MULTIPROCDS) $(MULTIPROCDH) $(MULTIPROCLIB) $(MULTIPROCMAP)

distclean::     distclean-$(MODNAME)
