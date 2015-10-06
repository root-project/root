# Module.mk for sqlite module
# Copyright (c) 2013 Rene Brun and Fons Rademakers
#
# Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

MODNAME      := sqlite
MODDIR       := $(ROOT_SRCDIR)/sql/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SQLITEDIR    := $(MODDIR)
SQLITEDIRS   := $(SQLITEDIR)/src
SQLITEDIRI   := $(SQLITEDIR)/inc

##### libSQLite #####
SQLITEL      := $(MODDIRI)/LinkDef.h
SQLITEDS     := $(call stripsrc,$(MODDIRS)/G__SQLite.cxx)
SQLITEDO     := $(SQLITEDS:.cxx=.o)
SQLITEDH     := $(SQLITEDS:.cxx=.h)

SQLITEH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SQLITES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SQLITEO      := $(call stripsrc,$(SQLITES:.cxx=.o))

SQLITEDEP     := $(SQLITEO:.o=.d) $(SQLITEDO:.o=.d)

SQLITELIB     := $(LPATH)/libRSQLite.$(SOEXT)
SQLITEMAP     := $(SQLITELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SQLITEH))
ALLLIBS     += $(SQLITELIB)
ALLMAPS     += $(SQLITEMAP)

# include all dependency files
INCLUDEFILES += $(SQLITEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SQLITEDIRI)/%.h
		cp $< $@

$(SQLITELIB):   $(SQLITEO) $(SQLITEDO) $(ORDER_) $(MAINLIBS) $(SQLITELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRSQLite.$(SOEXT) $@ "$(SQLITEO) $(SQLITEDO)" \
		   "$(SQLITELIBEXTRA) $(SQLITELIBDIR) $(SQLITECLILIB)"

$(SQLITEDS):    $(SQLITEH) $(SQLITEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SQLITEH) $(SQLITEL)

$(SQLITEMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(SQLITEL)
		$(RLIBMAP) -o $@ -l $(SQLITELIB) \
		   -d $(SQLITELIBDEPM) -c $(SQLITEL)

all-$(MODNAME): $(SQLITELIB) $(SQLITEMAP)

clean-$(MODNAME):
		@rm -f $(SQLITEO) $(SQLITEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SQLITEDEP) $(SQLITEDS) $(SQLITEDH) $(SQLITELIB) $(SQLITEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(SQLITEO) $(SQLITEDO): CXXFLAGS += $(SQLITEINCDIR:%=-I%)
