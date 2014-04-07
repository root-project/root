# Module.mk for sapdb module
# Copyright (c) 2001 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 24/8/2001

MODNAME      := sapdb
MODDIR       := $(ROOT_SRCDIR)/sql/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SAPDBDIR     := $(MODDIR)
SAPDBDIRS    := $(SAPDBDIR)/src
SAPDBDIRI    := $(SAPDBDIR)/inc

##### libSapDB #####
SAPDBL       := $(MODDIRI)/LinkDef.h
SAPDBDS      := $(call stripsrc,$(MODDIRS)/G__SapDB.cxx)
SAPDBDO      := $(SAPDBDS:.cxx=.o)
SAPDBDH      := $(SAPDBDS:.cxx=.h)

SAPDBH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SAPDBS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SAPDBO       := $(call stripsrc,$(SAPDBS:.cxx=.o))

SAPDBDEP     := $(SAPDBO:.o=.d) $(SAPDBDO:.o=.d)

SAPDBLIB     := $(LPATH)/libSapDB.$(SOEXT)
SAPDBMAP     := $(SAPDBLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SAPDBH))
ALLLIBS     += $(SAPDBLIB)
ALLMAPS     += $(SAPDBMAP)

# include all dependency files
INCLUDEFILES += $(SAPDBDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SAPDBDIRI)/%.h
		cp $< $@

$(SAPDBLIB):    $(SAPDBO) $(SAPDBDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSapDB.$(SOEXT) $@ "$(SAPDBO) $(SAPDBDO)" \
		   "$(SAPDBLIBEXTRA) $(SAPDBLIBDIR) $(SAPDBCLILIB)"

$(call pcmrule,SAPDB)
	$(noop)

$(SAPDBDS):     $(SAPDBH) $(SAPDBL) $(ROOTCLINGEXE) $(call pcmdep,SAPDB)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SAPDB) -c $(SAPDBH) $(SAPDBL)

$(SAPDBMAP):    $(SAPDBH) $(SAPDBL) $(ROOTCLINGEXE) $(call pcmdep,SAPDB)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SAPDBDS) $(call dictModule,SAPDB) -c $(SAPDBH) $(SAPDBL)

all-$(MODNAME): $(SAPDBLIB)

clean-$(MODNAME):
		@rm -f $(SAPDBO) $(SAPDBDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SAPDBDEP) $(SAPDBDS) $(SAPDBDH) $(SAPDBLIB) $(SAPDBMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(SAPDBO) $(SAPDBDO): CXXFLAGS += $(SAPDBINCDIR:%=-I%)
