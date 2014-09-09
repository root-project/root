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
LDAPDS       := $(call stripsrc,$(MODDIRS)/G__RLDAP.cxx)
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

$(call pcmrule,LDAP)
	$(noop)

$(LDAPDS):      $(LDAPH) $(LDAPL) $(ROOTCLINGEXE) $(call pcmdep,LDAP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,LDAP) -c $(LDAPH) $(LDAPL)

$(LDAPMAP):     $(LDAPH) $(LDAPL) $(ROOTCLINGEXE) $(call pcmdep,LDAP)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(LDAPDS) $(call dictModule,LDAP) -c $(LDAPH) $(LDAPL)

all-$(MODNAME): $(LDAPLIB)

clean-$(MODNAME):
		@rm -f $(LDAPO) $(LDAPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(LDAPDEP) $(LDAPDS) $(LDAPDH) $(LDAPLIB) $(LDAPMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(LDAPO): CXXFLAGS += -DLDAP_DEPRECATED $(LDAPINCDIR:%=-I%)
ifeq ($(MACOSX_LDAP_DEPRECATED),yes)
$(LDAPO) $(LDAPDO): CXXFLAGS += -Wno-deprecated-declarations
endif
