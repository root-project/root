# Module.mk for ldap module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/11/2002

MODDIR       := ldap
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

LDAPDIR      := $(MODDIR)
LDAPDIRS     := $(LDAPDIR)/src
LDAPDIRI     := $(LDAPDIR)/inc

##### libRLDAP #####
LDAPL        := $(MODDIRI)/LinkDef.h
LDAPDS       := $(MODDIRS)/G__LDAP.cxx
LDAPDO       := $(LDAPDS:.cxx=.o)
LDAPDH       := $(LDAPDS:.cxx=.h)

LDAPH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
LDAPS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
LDAPO        := $(LDAPS:.cxx=.o)

LDAPDEP      := $(LDAPO:.o=.d) $(LDAPDO:.o=.d)

LDAPLIB      := $(LPATH)/libRLDAP.$(SOEXT)
LDAPMAP      := $(LDAPLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(LDAPH))
ALLLIBS     += $(LDAPLIB)
ALLMAPS     += $(LDAPMAP)

# include all dependency files
INCLUDEFILES += $(LDAPDEP)

##### local rules #####
include/%.h:    $(LDAPDIRI)/%.h
		cp $< $@

$(LDAPLIB):     $(LDAPO) $(LDAPDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRLDAP.$(SOEXT) $@ "$(LDAPO) $(LDAPDO)" \
		   "$(LDAPLIBEXTRA) $(LDAPLIBDIR) $(LDAPCLILIB)"

$(LDAPDS):      $(LDAPH) $(LDAPL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(LDAPH) $(LDAPL)

$(LDAPMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(LDAPL)
		$(RLIBMAP) -o $(LDAPMAP) -l $(LDAPLIB) \
		   -d $(LDAPLIBDEPM) -c $(LDAPL)

all-ldap:       $(LDAPLIB) $(LDAPMAP)

clean-ldap:
		@rm -f $(LDAPO) $(LDAPDO)

clean::         clean-ldap

distclean-ldap: clean-ldap
		@rm -f $(LDAPDEP) $(LDAPDS) $(LDAPDH) $(LDAPLIB) $(LDAPMAP)

distclean::     distclean-ldap

##### extra rules ######
$(LDAPO): CXXFLAGS += -DLDAP_DEPRECATED $(LDAPINCDIR:%=-I%)
