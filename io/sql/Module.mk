# Module.mk for sql module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 7/12/2005

MODNAME      := sql
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SQLDIR       := $(MODDIR)
SQLDIRS      := $(SQLDIR)/src
SQLDIRI      := $(SQLDIR)/inc

##### libSQL #####
SQLL         := $(MODDIRI)/LinkDef.h
SQLDS        := $(call stripsrc,$(MODDIRS)/G__SQL.cxx)
SQLDO        := $(SQLDS:.cxx=.o)
SQLDH        := $(SQLDS:.cxx=.h)

SQLH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SQLS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SQLO         := $(call stripsrc,$(SQLS:.cxx=.o))

SQLDEP       := $(SQLO:.o=.d) $(SQLDO:.o=.d)

SQLLIB       := $(LPATH)/libSQLIO.$(SOEXT)
SQLMAP       := $(SQLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SQLH))
ALLLIBS      += $(SQLLIB)
ALLMAPS      += $(SQLMAP)

# include all dependency files
INCLUDEFILES += $(SQLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SQLDIRI)/%.h
		cp $< $@

$(SQLLIB):      $(SQLO) $(SQLDO) $(ORDER_) $(MAINLIBS) $(SQLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSQLIO.$(SOEXT) $@ "$(SQLO) $(SQLDO)" \
		   "$(SQLLIBEXTRA)"

$(SQLDS):       $(SQLH) $(SQLL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SQLH) $(SQLL)

$(SQLMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(SQLL)
		$(RLIBMAP) -o $@ -l $(SQLLIB) \
		   -d $(SQLLIBDEPM) -c $(SQLL)

all-$(MODNAME): $(SQLLIB) $(SQLMAP)

clean-$(MODNAME):
		@rm -f $(SQLO) $(SQLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SQLDEP) $(SQLDS) $(SQLDH) $(SQLLIB) $(SQLMAP)

distclean::     distclean-$(MODNAME)
