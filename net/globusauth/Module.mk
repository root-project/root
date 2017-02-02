# Module.mk for krb5 authentication module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/3/2002 (for krb5auth)
# Mod by: Gerardo Ganis, 18/1/2003

MODNAME      := globusauth
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLBSAUTHDIR  := $(MODDIR)
GLBSAUTHDIRS := $(GLBSAUTHDIR)/src
GLBSAUTHDIRI := $(GLBSAUTHDIR)/inc

##### libGlobusAuth #####
GLBSAUTHH    := $(wildcard $(MODDIRI)/*.h)
GLBSAUTHS    := $(wildcard $(MODDIRS)/*.cxx)
GLBSAUTHO    := $(call stripsrc,$(GLBSAUTHS:.cxx=.o))

GLBSAUTHDEP  := $(GLBSAUTHO:.o=.d)

GLBSAUTHLIB  := $(LPATH)/libGlobusAuth.$(SOEXT)

##### experimental patch #####
GLBPATCHS     :=
GLBPATCHO     :=
GLBPATCHDEP   :=
ifneq ($(GLBPATCHFLAGS),)
GLBPATCHS     := $(MODDIRS)/globus_gsi_credential.c
GLBPATCHO     := $(call stripsrc,$(GLBPATCHS:.c=.o))
GLBPATCHDEP   := $(GLBPATCHO:.o=.d)
endif

# used in the main Makefile
GLBSAUTHH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLBSAUTHH))
ALLHDRS     += $(GLBSAUTHH_REL)
ALLLIBS     += $(GLBSAUTHLIB)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(GLBSAUTHH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(GLBSAUTHLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(GLBSAUTHDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GLBSAUTHDIRI)/%.h
		cp $< $@

$(GLBSAUTHLIB): $(GLBSAUTHO) $(GLBPATCHO) $(ORDER_) $(MAINLIBS) $(GLBSAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGlobusAuth.$(SOEXT) $@ \
		   "$(GLBSAUTHO) $(GLBPATCHO)" \
		   "$(GLBSAUTHLIBEXTRA) $(GLOBUSLIBDIR) $(GLOBUSLIB)"

all-$(MODNAME): $(GLBSAUTHLIB)

clean-$(MODNAME):
		@rm -f $(GLBSAUTHO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GLBSAUTHDEP) $(GLBSAUTHLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GLBSAUTHO): CXXFLAGS += $(GLOBUSINCDIR:%=-I%)
$(GLBPATCHO): CFLAGS += $(GLBPATCHFLAGS) $(GLOBUSINCDIR:%=-I%)
