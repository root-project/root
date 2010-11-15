# Module.mk for srputils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := srputils
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SRPUTILSDIR  := $(MODDIR)
SRPUTILSDIRS := $(SRPUTILSDIR)/src
SRPUTILSDIRI := $(SRPUTILSDIR)/inc

##### libSRPAuth #####
SRPUTILSH    := $(wildcard $(MODDIRI)/*.h)
SRPUTILSS    := $(wildcard $(MODDIRS)/*.cxx)
SRPUTILSO    := $(call stripsrc,$(SRPUTILSS:.cxx=.o))

SRPUTILSDEP  := $(SRPUTILSO:.o=.d)

SRPUTILSLIB  := $(LPATH)/libSRPAuth.$(SOEXT)

##### rpasswd #####
RPASSWDS     := $(MODDIRS)/rpasswd.c
RPASSWDO     := $(call stripsrc,$(RPASSWDS:.c=.o))
RPASSWDDEP   := $(RPASSWDO:.o=.d)
RPASSWD      := bin/rpasswd

##### rtconf #####
RTCONFS      := $(MODDIRS)/rtconf.c
RTCONFO      := $(call stripsrc,$(RTCONFS:.c=.o))
RTCONFDEP    := $(RTCONFO:.o=.d)
RTCONF       := bin/rtconf

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SRPUTILSH))
ALLLIBS     += $(SRPUTILSLIB)
ALLEXECS    += $(RPASSWD) $(RTCONF)

# include all dependency files
INCLUDEFILES += $(SRPUTILSDEP) $(RPASSWDDEP) $(RTCONFDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SRPUTILSDIRI)/%.h
		cp $< $@

$(SRPUTILSLIB): $(SRPUTILSO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSRPAuth.$(SOEXT) $@ "$(SRPUTILSO)" \
		   "$(SRPUTILSLIBEXTRA) $(SRPLIBDIR) $(SRPLIB) \
		    $(CRYPTOLIBDIR) $(CRYPTOLIB)"

$(RPASSWD):     $(RPASSWDO)
		$(LD) $(LDFLAGS) -o $@ $(RPASSWDO) \
		   $(SRPUTILLIBDIR) $(SRPLIBDIR) $(SRPUTILLIB) $(SRPLIB) \
		   $(CRYPTOLIBDIR) $(CRYPTOLIB) -lcrack

$(RTCONF):      $(RTCONFO)
		$(LD) $(LDFLAGS) -o $@ $(RTCONFO) \
		   $(SRPLIBDIR) $(SRPLIB) $(CRYPTOLIBDIR) $(CRYPTOLIB)

all-$(MODNAME): $(SRPUTILSLIB) $(RPASSWD) $(RTCONF)

clean-$(MODNAME):
		@rm -f $(SRPUTILSO) $(RPASSWDO) $(RTCONFO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SRPUTILSDEP) $(SRPUTILSLIB) $(RPASSWDDEP) $(RPASSWD) \
		   $(RTCONFDEP) $(RTCONF)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(SRPUTILSO): CXXFLAGS += $(SRPINCDIR:%=-I%)
$(RTCONFO): CFLAGS += $(SRPUTILINCDIR:%=-I%) $(SRPINCDIR:%=-I%)
