# Module.mk for srputils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := srputils
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SRPUTILSDIR  := $(MODDIR)
SRPUTILSDIRS := $(SRPUTILSDIR)/src
SRPUTILSDIRI := $(SRPUTILSDIR)/inc

##### libSRPAuth #####
SRPUTILSH    := $(wildcard $(MODDIRI)/*.h)
SRPUTILSS    := $(wildcard $(MODDIRS)/*.cxx)
SRPUTILSO    := $(SRPUTILSS:.cxx=.o)

SRPUTILSDEP  := $(SRPUTILSO:.o=.d)

SRPUTILSLIB  := $(LPATH)/libSRPAuth.$(SOEXT)

##### rpasswd #####
RPASSWDS     := $(MODDIRS)/rpasswd.c
RPASSWDO     := $(RPASSWDS:.c=.o)
RPASSWDDEP   := $(RPASSWDO:.o=.d)
RPASSWD      := bin/rpasswd

##### rtconf #####
RTCONFS      := $(MODDIRS)/rtconf.c
RTCONFO      := $(RTCONFS:.c=.o)
RTCONFDEP    := $(RTCONFO:.o=.d)
RTCONF       := bin/rtconf

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SRPUTILSH))
ALLLIBS     += $(SRPUTILSLIB)
ALLEXECS    += $(RPASSWD) $(RTCONF)

# include all dependency files
INCLUDEFILES += $(SRPUTILSDEP) $(RPASSWDDEP) $(RTCONFDEP)

##### local rules #####
include/%.h:    $(SRPUTILSDIRI)/%.h
		cp $< $@

$(SRPUTILSLIB): $(SRPUTILSO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSRPAuth.$(SOEXT) $@ "$(SRPUTILSO)" \
		   "$(SRPUTILSLIBEXTRA) $(SRPLIBDIR) $(SRPLIB)"

$(RPASSWD):     $(RPASSWDO)
		$(LD) $(LDFLAGS) -o $@ $(RPASSWDO) \
		   $(SRPUTILLIBDIR) $(SRPLIBDIR) $(SRPUTILLIB) $(SRPLIB) -lcrack

$(RTCONF):      $(RTCONFO)
		$(LD) $(LDFLAGS) -o $@ $(RTCONFO) \
		   $(SRPLIBDIR) $(SRPLIB)

all-srputils:   $(SRPUTILSLIB) $(RPASSWD) $(RTCONF)

clean-srputils:
		@rm -f $(SRPUTILSO) $(RPASSWDO) $(RTCONFO)

clean::         clean-srputils

distclean-srputils: clean-srputils
		@rm -f $(SRPUTILSDEP) $(SRPUTILSLIB) $(RPASSWDDEP) $(RPASSWD) \
		   $(RTCONFDEP) $(RTCONF)

distclean::     distclean-srputils

##### extra rules ######
$(SRPUTILSO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(SRPINCDIR) -o $@ -c $<

$(RPASSWDO): $(RPASSWDS)
	$(CC) $(OPT) $(CFLAGS) -I$(SRPUTILINCDIR) -I$(SRPINCDIR) \
	   -o $@ -c $<

$(RTCONFO): $(RTCONFS)
	$(CC) $(OPT) $(CFLAGS) -I$(SRPUTILINCDIR) -I$(SRPINCDIR) \
	   -o $@ -c $<
