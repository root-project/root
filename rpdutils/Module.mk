# Module.mk for Rootd/Proofd authentication utilities
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Gerardo Ganis, 7/4/2003

MODDIR       := rpdutils
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RPDUTILDIR   := $(MODDIR)
RPDUTILDIRS  := $(RPDUTILDIR)/src
RPDUTILDIRI  := $(RPDUTILDIR)/inc

##### $(RPDUTILO) #####
RPDUTILH     := $(wildcard $(MODDIRI)/*.h)
RPDUTILS     := $(wildcard $(MODDIRS)/*.cxx)
RPDUTILO     := $(RPDUTILS:.cxx=.o)

RPDUTILDEP   := $(RPDUTILO:.o=.d)

##### Flags used in rootd amd proofd Module.mk #####
# use shadow passwords for authentication
ifneq ($(SHADOWFLAGS),)
SHADOWLIBS   := $(SHADOWLIBDIR) $(SHADOWLIB)
endif

# use AFS for authentication
ifneq ($(AFSLIB),)
AFSFLAGS     := -DR__AFS
AFSLIBS      := $(AFSLIBDIR) $(AFSLIB)
endif

# use SRP for authentication
ifneq ($(SRPLIB),)
SRPFLAGS     := -I$(SRPINCDIR)
SRPLIBS      := $(SRPLIBDIR) $(SRPLIB)
endif

# use krb5 for authentication
ifneq ($(KRB5LIB),)
KRB5FLAGS     := -I$(KRB5INCDIR)
KRB5LIBS      := $(KRB5LIBDIR) $(KRB5LIB)
endif

# use Globus for authentication
ifneq ($(GLOBUSLIB),)
GLBSFLAGS     := -I$(GLOBUSINCDIR)
GLBSLIBS      := $(GLOBUSLIBDIR) $(GLOBUSLIB)
else
RPDUTILS      := $(filter-out $(MODDIRS)/globus.cxx,$(RPDUTILS))
RPDUTILO      := $(filter-out $(MODDIRS)/globus.o,$(RPDUTILO))
endif

# Combined...
AUTHFLAGS     := $(EXTRA_AUTHFLAGS) $(SHADOWFLAGS) $(AFSFLAGS) $(SRPFLAGS) \
                 $(KRB5FLAGS) $(GLBSFLAGS) $(SSLFLAGS)
AUTHLIBS      := $(SHADOWLIBS) $(AFSLIBS) $(SRPLIBS) $(KRB5LIBS) $(GLBSLIBS) \
                 $(COMERRLIBDIR) $(COMERRLIB) $(RESOLVLIB) \
                 $(CRYPTOLIBDIR) $(CRYPTOLIB)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RPDUTILH))

# include all dependency files
INCLUDEFILES  += $(RPDUTILDEP)

##### local rules #####
include/%.h:    $(RPDUTILDIRI)/%.h
		cp $< $@

all-rpdutils:   $(RPDUTILO)

clean-rpdutils:
		@rm -f $(RPDUTILO)

clean::         clean-rpdutils

distclean-rpdutils: clean-rpdutils
		@rm -f $(RPDUTILDEP)

distclean::     distclean-rpdutils

##### extra rules ######
$(RPDUTILO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(AUTHFLAGS) -o $@ -c $<
