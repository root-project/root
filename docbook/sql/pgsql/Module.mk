# Module.mk for pgsql module
# Copyright (c) 2001 Rene Brun and Fons Rademakers
#
# Author: g.p.ciceri <gp.ciceri@acm.org>, 1/06/2001

MODNAME      := pgsql
MODDIR       := sql/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PGSQLDIR     := $(MODDIR)
PGSQLDIRS    := $(PGSQLDIR)/src
PGSQLDIRI    := $(PGSQLDIR)/inc

##### libPgSQL #####
PGSQLL       := $(MODDIRI)/LinkDef.h
PGSQLDS      := $(MODDIRS)/G__PgSQL.cxx
PGSQLDO      := $(PGSQLDS:.cxx=.o)
PGSQLDH      := $(PGSQLDS:.cxx=.h)

PGSQLH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PGSQLS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PGSQLO       := $(PGSQLS:.cxx=.o)

PGSQLDEP     := $(PGSQLO:.o=.d) $(PGSQLDO:.o=.d)

PGSQLLIB     := $(LPATH)/libPgSQL.$(SOEXT)
PGSQLMAP     := $(PGSQLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PGSQLH))
ALLLIBS     += $(PGSQLLIB)
ALLMAPS     += $(PGSQLMAP)

# include all dependency files
INCLUDEFILES += $(PGSQLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PGSQLDIRI)/%.h
		cp $< $@

$(PGSQLLIB):    $(PGSQLO) $(PGSQLDO) $(ORDER_) $(MAINLIBS) $(PGSQLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPgSQL.$(SOEXT) $@ "$(PGSQLO) $(PGSQLDO)" \
		   "$(PGSQLLIBEXTRA) $(PGSQLLIBDIR) $(PGSQLCLILIB)"

$(PGSQLDS):     $(PGSQLH) $(PGSQLL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PGSQLH) $(PGSQLL)

$(PGSQLMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(PGSQLL)
		$(RLIBMAP) -o $@ -l $(PGSQLLIB) \
		   -d $(PGSQLLIBDEPM) -c $(PGSQLL)

all-$(MODNAME): $(PGSQLLIB) $(PGSQLMAP)

clean-$(MODNAME):
		@rm -f $(PGSQLO) $(PGSQLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PGSQLDEP) $(PGSQLDS) $(PGSQLDH) $(PGSQLLIB) $(PGSQLMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PGSQLO) $(PGSQLDO): CXXFLAGS += $(PGSQLINCDIR:%=-I%)
