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
ODBCMAP      := $(ODBCLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ODBCH))
ALLLIBS     += $(ODBCLIB)
ALLMAPS     += $(ODBCMAP)

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

$(ODBCMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(ODBCL)
		$(RLIBMAP) -o $(ODBCMAP) -l $(ODBCLIB) \
		   -d $(ODBCLIBDEPM) -c $(ODBCL)

all-odbc:      $(ODBCLIB) $(ODBCMAP)

clean-odbc:
		@rm -f $(ODBCO) $(ODBCDO)

clean::         clean-odbc

distclean-odbc: clean-odbc
		@rm -f $(ODBCDEP) $(ODBCDS) $(ODBCDH) $(ODBCLIB) $(ODBCMAP)

distclean::     distclean-odbc

##### extra rules ######
$(ODBCO) $(ODBCDO): CXXFLAGS += $(ODBCINCDIR:%=-I%)
