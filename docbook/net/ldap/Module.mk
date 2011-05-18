# Module.mk for ldap module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/11/2002

MODNAME      := ldap
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

LDAPDIR      := $(MODDIR)
LDAPDIRS     := $(LDAPDIR)/src
LDAPDIRI     := $(LDAPDIR)/inc

##### libRLDAP #####
LDAPL        := $(MODDIRI)/LinkDef.h
LDAPDS       := $(call stripsrc,$(MODDIRS)/G__LDAP.cxx)
LDAPDO       := $(LDAPDS:.cxx=.o)
LDAPDH       := $(LDAPDS:.cxx=.h)

LDAPH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
LDAPS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
LDAPO        := $(call stripsrc,$(LDAPS:.cxx=.o))

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
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(LDAPDIRI)/%.h
		cp $< $@

$(LDAPLIB):     $(LDAPO) $(LDAPDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRLDAP.$(SOEXT) $@ "$(LDAPO) $(LDAPDO)" \
		   "$(LDAPLIBEXTRA) $(LDAPLIBDIR) $(LDAPCLILIB)"

$(LDAPDS):      $(LDAPH) $(LDAPL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(LDAPH) $(LDAPL)

$(LDAPMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(LDAPL)
		$(RLIBMAP) -o $@ -l $(LDAPLIB) \
		   -d $(LDAPLIBDEPM) -c $(LDAPL)

all-$(MODNAME): $(LDAPLIB) $(LDAPMAP)

clean-$(MODNAME):
		@rm -f $(LDAPO) $(LDAPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(LDAPDEP) $(LDAPDS) $(LDAPDH) $(LDAPLIB) $(LDAPMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(LDAPO): CXXFLAGS += -DLDAP_DEPRECATED $(LDAPINCDIR:%=-I%)
