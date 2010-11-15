# Module.mk for table module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := table
MODDIR       := $(ROOT_SRCDIR)/misc/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TABLEDIR     := $(MODDIR)
TABLEDIRS    := $(TABLEDIR)/src
TABLEDIRI    := $(TABLEDIR)/inc

##### libTable #####
TABLEL       := $(MODDIRI)/LinkDef.h
TABLEDS      := $(call stripsrc,$(MODDIRS)/G__Table.cxx)
TABLEDO      := $(TABLEDS:.cxx=.o)
TABLEDH      := $(TABLEDS:.cxx=.h)

TABLEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TABLES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TABLEO       := $(call stripsrc,$(TABLES:.cxx=.o))

TABLEDEP     := $(TABLEO:.o=.d) $(TABLEDO:.o=.d)

TABLELIB     := $(LPATH)/libTable.$(SOEXT)
TABLEMAP     := $(TABLELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TABLEH))
ALLLIBS     += $(TABLELIB)
ALLMAPS     += $(TABLEMAP)

# include all dependency files
INCLUDEFILES += $(TABLEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TABLEDIRI)/%.h
		cp $< $@

$(TABLELIB):    $(TABLEO) $(TABLEDO) $(ORDER_) $(MAINLIBS) $(TABLELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTable.$(SOEXT) $@ "$(TABLEO) $(TABLEDO)" \
		   "$(TABLELIBEXTRA)"

$(TABLEDS):     $(TABLEH) $(TABLEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TABLEH) $(TABLEL)

$(TABLEMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(TABLEL)
		$(RLIBMAP) -o $@ -l $(TABLELIB) \
		   -d $(TABLELIBDEPM) -c $(TABLEL)

all-$(MODNAME): $(TABLELIB) $(TABLEMAP)

clean-$(MODNAME):
		@rm -f $(TABLEO) $(TABLEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TABLEDEP) $(TABLEDS) $(TABLEDH) $(TABLELIB) $(TABLEMAP)

distclean::     distclean-$(MODNAME)

$(TABLEO) $(TABLEDO): CXXFLAGS += -Iinclude/cint
