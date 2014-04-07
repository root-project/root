# Module.mk for pgsql module
# Copyright (c) 2001 Rene Brun and Fons Rademakers
#
# Author: g.p.ciceri <gp.ciceri@acm.org>, 1/06/2001

MODNAME      := pgsql
MODDIR       := $(ROOT_SRCDIR)/sql/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PGSQLDIR     := $(MODDIR)
PGSQLDIRS    := $(PGSQLDIR)/src
PGSQLDIRI    := $(PGSQLDIR)/inc

##### libPgSQL #####
PGSQLL       := $(MODDIRI)/LinkDef.h
PGSQLDS      := $(call stripsrc,$(MODDIRS)/G__PgSQL.cxx)
PGSQLDO      := $(PGSQLDS:.cxx=.o)
PGSQLDH      := $(PGSQLDS:.cxx=.h)

PGSQLH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PGSQLS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PGSQLO       := $(call stripsrc,$(PGSQLS:.cxx=.o))

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

$(call pcmrule,PGSQL)
	$(noop)

$(PGSQLDS):     $(PGSQLH) $(PGSQLL) $(ROOTCLINGEXE) $(call pcmdep,PGSQL)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PGSQL) -c $(PGSQLINCDIR:%=-I%) $(PGSQLH) $(PGSQLL)

$(PGSQLMAP):    $(PGSQLH) $(PGSQLL) $(ROOTCLINGEXE) $(call pcmdep,PGSQL)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(PGSQLDS) $(call dictModule,PGSQL) -c $(PGSQLINCDIR:%=-I%) $(PGSQLH) $(PGSQLL)

all-$(MODNAME): $(PGSQLLIB)

clean-$(MODNAME):
		@rm -f $(PGSQLO) $(PGSQLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PGSQLDEP) $(PGSQLDS) $(PGSQLDH) $(PGSQLLIB) $(PGSQLMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PGSQLO) $(PGSQLDO): CXXFLAGS += $(PGSQLINCDIR:%=-I%)
