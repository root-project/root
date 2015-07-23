# Module.mk for hbook module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/2/2002

MODNAME      := hbook
MODDIR       := $(ROOT_SRCDIR)/hist/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HBOOKDIR     := $(MODDIR)
HBOOKDIRS    := $(HBOOKDIR)/src
HBOOKDIRI    := $(HBOOKDIR)/inc

##### libHbook #####
HBOOKL       := $(MODDIRI)/LinkDef.h
HBOOKDS      := $(call stripsrc,$(MODDIRS)/G__Hbook.cxx)
HBOOKDO      := $(HBOOKDS:.cxx=.o)
HBOOKDH      := $(HBOOKDS:.cxx=.h)

HBOOKH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HBOOKS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HBOOKO       := $(call stripsrc,$(HBOOKS:.cxx=.o))

HBOOKDEP     := $(HBOOKO:.o=.d) $(HBOOKDO:.o=.d)

HBOOKLIB     := $(LPATH)/libHbook.$(SOEXT)
HBOOKMAP     := $(HBOOKLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HBOOKH))
ALLLIBS     += $(HBOOKLIB)
ALLMAPS     += $(HBOOKMAP)

# include all dependency files
INCLUDEFILES += $(HBOOKDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HBOOKDIRI)/%.h
		cp $< $@

$(HBOOKLIB):    $(HBOOKO) $(HBOOKDO) $(ORDER_) $(MAINLIBS) $(HBOOKLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHbook.$(SOEXT) $@ "$(HBOOKO) $(HBOOKDO)" \
		   "$(HBOOKLIBEXTRA) $(F77LIBS)"

$(call pcmrule,HBOOK)
	$(noop)

$(HBOOKDS):     $(HBOOKH) $(HBOOKL) $(ROOTCLINGEXE) $(filter-out lib/libminicern_rdict.pcm,$(call pcmdep,HBOOK))
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,HBOOK) -c -writeEmptyRootPCM $(HBOOKH) $(HBOOKL)

$(HBOOKMAP):    $(HBOOKH) $(HBOOKL) $(ROOTCLINGEXE) $(filter-out lib/libminicern_rdict.pcm,$(call pcmdep,HBOOK))
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HBOOKDS) $(call dictModule,HBOOK) -c $(HBOOKH) $(HBOOKL)

all-$(MODNAME): $(HBOOKLIB)

clean-$(MODNAME):
		@rm -f $(HBOOKO) $(HBOOKDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HBOOKDEP) $(HBOOKDS) $(HBOOKDH) $(HBOOKLIB) $(HBOOKMAP)

distclean::     distclean-$(MODNAME)
