# Module.mk for rootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := rootd
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTDDIR     := $(MODDIR)
ROOTDDIRS    := $(ROOTDDIR)/src
ROOTDDIRI    := $(ROOTDDIR)/inc

##### rootd #####
ROOTDH       := $(wildcard $(MODDIRI)/*.h)
ROOTDS       := $(wildcard $(MODDIRS)/*.cxx)
ROOTDO       := $(ROOTDS:.cxx=.o)
ROOTDDEP     := $(ROOTDO:.o=.d)
ROOTD        := bin/rootd

##### use shadow passwords for authentication #####
ifneq ($(SHADOWFLAGS),)
SHADOWLIBS   := $(SHADOWLIBDIR) $(SHADOWLIB)
endif

##### use AFS for authentication #####
ifneq ($(AFSLIB),)
AFSFLAGS     := -DR__AFS
AFSLIBS      := $(AFSLIBDIR) $(AFSLIB)
endif

##### use SRP for authentication #####
ifneq ($(SRPLIB),)
SRPFLAGS     := -DR__SRP -I$(SRPINCDIR)
SRPLIBS      := $(SRPLIBDIR) $(SRPLIB)
endif

##### use krb5 for authentication #####
ifneq ($(KRB5LIB),)
KRB5FLAGS     := -DR__KRB5 -I$(KRB5INCDIR)
KRB5LIBS      := $(KRB5LIBDIR) $(KRB5LIB)
endif

AUTHFLAGS    := $(SHADOWFLAGS) $(AFSFLAGS) $(SRPFLAGS) $(KRB5FLAGS) \
                $(EXTRA_AUTHFLAGS)
AUTHLIBS     := $(SHADOWLIBS) $(AFSLIBS) $(SRPLIBS) $(KRB5LIBS)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTDH))
ALLEXECS     += $(ROOTD)

# include all dependency files
INCLUDEFILES += $(ROOTDDEP)

##### local rules #####
include/%.h:    $(ROOTDDIRI)/%.h
		cp $< $@

$(ROOTD):       $(ROOTDO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTDO) $(AUTHLIBS) $(CRYPTLIBS) \
		   $(SYSLIBS)

all-rootd:      $(ROOTD)

clean-rootd:
		@rm -f $(ROOTDO)

clean::         clean-rootd

distclean-rootd: clean-rootd
		@rm -f $(ROOTDDEP) $(ROOTD)

distclean::     distclean-rootd

##### extra rules ######
$(ROOTDDIRS)/rootd.o: $(ROOTDDIRS)/rootd.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(AUTHFLAGS) -o $@ -c $<
