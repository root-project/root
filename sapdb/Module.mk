# Module.mk for sapdb module
# Copyright (c) 2001 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 24/8/2001

MODDIR       := sapdb
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SAPDBDIR     := $(MODDIR)
SAPDBDIRS    := $(SAPDBDIR)/src
SAPDBDIRI    := $(SAPDBDIR)/inc

##### libSapDB #####
SAPDBL       := $(MODDIRI)/LinkDef.h
SAPDBDS      := $(MODDIRS)/G__SapDB.cxx
SAPDBDO      := $(SAPDBDS:.cxx=.o)
SAPDBDH      := $(SAPDBDS:.cxx=.h)

SAPDBH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SAPDBS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SAPDBO       := $(SAPDBS:.cxx=.o)

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
include/%.h:    $(SAPDBDIRI)/%.h
		cp $< $@

$(SAPDBLIB):    $(SAPDBO) $(SAPDBDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSapDB.$(SOEXT) $@ "$(SAPDBO) $(SAPDBDO)" \
		   "$(SAPDBLIBEXTRA) $(SAPDBLIBDIR) $(SAPDBCLILIB)"

$(SAPDBDS):     $(SAPDBH) $(SAPDBL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SAPDBH) $(SAPDBL)

$(SAPDBMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(SAPDBL)
		$(RLIBMAP) -o $(SAPDBMAP) -l $(SAPDBLIB) \
		   -d $(SAPDBLIBDEPM) -c $(SAPDBL)

all-sapdb:      $(SAPDBLIB) $(SAPDBMAP)

clean-sapdb:
		@rm -f $(SAPDBO) $(SAPDBDO)

clean::         clean-sapdb

distclean-sapdb: clean-sapdb
		@rm -f $(SAPDBDEP) $(SAPDBDS) $(SAPDBDH) $(SAPDBLIB) $(SAPDBMAP)

distclean::     distclean-sapdb

##### extra rules ######
$(SAPDBO) $(SAPDBDO): CXXFLAGS += $(SAPDBINCDIR:%=-I%)
