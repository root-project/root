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

MULTIPROCH      := $(MODDIRI)/TMPClient.h $(MODDIRI)/TProcPool.h \
                $(MODDIRI)/TMPWorker.h $(MODDIRI)/MPSendRecv.h \
                $(MODDIRI)/TPoolWorker.h $(MODDIRI)/MPCode.h \
                $(MODDIRI)/PoolCode.h

MULTIPROCS      := $(MODDIRS)/TMPClient.cxx $(MODDIRS)/TProcPool.cxx \
                $(MODDIRS)/TMPWorker.cxx $(MODDIRS)/MPSendRecv.cxx \
                $(MODDIRS)/TPoolWorker.cxx

MULTIPROCO      := $(call stripsrc,$(MULTIPROCS:.cxx=.o))

MULTIPROCDEP    := $(MULTIPROCO:.o=.d) $(MULTIPROCDO:.o=.d)

MULTIPROCLIB    := $(LPATH)/libMultiProc.$(SOEXT)
MULTIPROCMAP    := $(MULTIPROCLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MULTIPROCH) $(MULTIPROCH_EXT))
ALLLIBS      += $(MULTIPROCLIB)
ALLMAPS      += $(MULTIPROCMAP)

CXXFLAGS     += $(OSMULTIPROCFLAG)
CFLAGS       += $(OSMULTIPROCFLAG)
CINTCXXFLAGS += $(OSMULTIPROCFLAG)
CINTCFLAGS   += $(OSMULTIPROCFLAG)

# include all dependency files
INCLUDEFILES += $(MULTIPROCDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MULTIPROCDIRI)/%.h
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

