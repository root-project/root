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

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PGSQLH))
ALLLIBS     += $(PGSQLLIB)

# include all dependency files
INCLUDEFILES += $(PGSQLDEP)

##### local rules #####
include/%.h:    $(PGSQLDIRI)/%.h
		cp $< $@

$(PGSQLLIB):    $(PGSQLO) $(PGSQLDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPgSQL.$(SOEXT) $@ "$(PGSQLO) $(PGSQLDO)" \
		   "$(PGSQLLIBEXTRA) $(PGSQLLIBDIR) $(PGSQLCLILIB)"

$(PGSQLDS):     $(PGSQLH) $(PGSQLL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PGSQLH) $(PGSQLL)

$(PGSQLDO):     $(PGSQLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(PGSQLINCDIR:%=-I%) -I. -o $@ -c $<

all-pgsql:      $(PGSQLLIB)

map-pgsql:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PGSQLLIB) \
		   -d $(PGSQLLIBDEP) -c $(PGSQLL)

map::           map-pgsql

clean-pgsql:
		@rm -f $(PGSQLO) $(PGSQLDO)

clean::         clean-pgsql

distclean-pgsql: clean-pgsql
		@rm -f $(PGSQLDEP) $(PGSQLDS) $(PGSQLDH) $(PGSQLLIB)

distclean::     distclean-pgsql

##### extra rules ######
$(PGSQLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(PGSQLINCDIR:%=-I%) -o $@ -c $<
