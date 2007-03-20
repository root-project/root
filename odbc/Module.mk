# Module.mk for odbc module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR      := odbc
MODDIRS     := $(MODDIR)/src
MODDIRI     := $(MODDIR)/inc

ODBCDIR     := $(MODDIR)
ODBCDIRS    := $(ODBCDIR)/src
ODBCDIRI    := $(ODBCDIR)/inc

##### libODBC #####
ODBCL       := $(MODDIRI)/LinkDef.h
ODBCDS      := $(MODDIRS)/G__ODBC.cxx
ODBCDO      := $(ODBCDS:.cxx=.o)
ODBCDH      := $(ODBCDS:.cxx=.h)

ODBCH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ODBCS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ODBCO       := $(ODBCS:.cxx=.o)

ODBCDEP     := $(ODBCO:.o=.d) $(ODBCDO:.o=.d)

ODBCLIB     := $(LPATH)/libRODBC.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ODBCH))
ALLLIBS     += $(ODBCLIB)

# include all dependency files
INCLUDEFILES += $(ODBCDEP)

##### local rules #####
include/%.h:    $(ODBCDIRI)/%.h
		cp $< $@

$(ODBCLIB):     $(ODBCO) $(ODBCDO) $(ORDER_) $(MAINLIBS) $(ODBCLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRODBC.$(SOEXT) $@ "$(ODBCO) $(ODBCDO)" \
		   "$(ODBCLIBEXTRA) $(ODBCLIBDIR) $(ODBCCLILIB)"

$(ODBCDS):     $(ODBCH) $(ODBCL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ODBCINCDIR:%=-I%) $(ODBCH) $(ODBCL)

all-odbc:      $(ODBCLIB)

map-odbc:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ODBCLIB) \
		   -d $(ODBCLIBDEP) -c $(ODBCL)

map::           map-odbc

clean-odbc:
		@rm -f $(ODBCO) $(ODBCDO)

clean::         clean-odbc

distclean-odbc: clean-odbc
		@rm -f $(ODBCDEP) $(ODBCDS) $(ODBCDH) $(ODBCLIB)

distclean::     distclean-odbc

##### extra rules ######
$(ODBCO) $(ODBCDO): CXXFLAGS += $(ODBCINCDIR:%=-I%)
