# Module.mk for pgsql module
# Copyright (c) 2001 Rene Brun and Fons Rademakers
#
# Author: g.p.ciceri <gp.ciceri@acm.org>, 1/06/2001

MODDIR       := pgsql
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
include/%.h:    $(PGSQLDIRI)/%.h
		cp $< $@

$(PGSQLLIB):    $(PGSQLO) $(PGSQLDO) $(ORDER_) $(MAINLIBS) $(PGSQLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPgSQL.$(SOEXT) $@ "$(PGSQLO) $(PGSQLDO)" \
		   "$(PGSQLLIBEXTRA) $(PGSQLLIBDIR) $(PGSQLCLILIB)"

$(PGSQLDS):     $(PGSQLH) $(PGSQLL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PGSQLH) $(PGSQLL)

$(PGSQLMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(PGSQLL)
		$(RLIBMAP) -o $(PGSQLMAP) -l $(PGSQLLIB) \
		   -d $(PGSQLLIBDEPM) -c $(PGSQLL)

all-pgsql:      $(PGSQLLIB) $(PGSQLMAP)

clean-pgsql:
		@rm -f $(PGSQLO) $(PGSQLDO)

clean::         clean-pgsql

distclean-pgsql: clean-pgsql
		@rm -f $(PGSQLDEP) $(PGSQLDS) $(PGSQLDH) $(PGSQLLIB) $(PGSQLMAP)

distclean::     distclean-pgsql

##### extra rules ######
$(PGSQLO) $(PGSQLDO): CXXFLAGS += $(PGSQLINCDIR:%=-I%)
