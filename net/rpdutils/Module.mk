# Module.mk for Rootd/Proofd authentication utilities
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Gerardo Ganis, 7/4/2003

MODNAME      := rpdutils
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RPDUTILDIR   := $(MODDIR)
RPDUTILDIRS  := $(RPDUTILDIR)/src
RPDUTILDIRI  := $(RPDUTILDIR)/inc

##### $(RPDUTILO) #####
RPDUTILH     := $(filter-out $(MODDIRI)/rpdpriv.h, $(filter-out $(MODDIRI)/rpdconn.h, $(wildcard $(MODDIRI)/*.h)))
RPDUTILS     := $(filter-out $(MODDIRS)/rpdpriv.cxx, $(filter-out $(MODDIRS)/rpdconn.cxx, $(wildcard $(MODDIRS)/*.cxx)))
RPDUTILO     := $(call stripsrc,$(RPDUTILS:.cxx=.o))

RPDUTILDEP   := $(RPDUTILO:.o=.d)

##### $(RPDCONNO) #####
RPDCONNH     := $(MODDIRI)/rpdconn.h
RPDCONNS     := $(MODDIRS)/rpdconn.cxx
RPDCONNO     := $(call stripsrc,$(RPDCONNS:.cxx=.o))

RPDCONNDEP   := $(RPDCONNO:.o=.d)

##### $(RPDPRIVO) #####
RPDPRIVH     := $(MODDIRI)/rpdpriv.h
RPDPRIVS     := $(MODDIRS)/rpdpriv.cxx
RPDPRIVO     := $(call stripsrc,$(RPDPRIVS:.cxx=.o))

RPDPRIVDEP   := $(RPDPRIVO:.o=.d)

##### for libSrvAuth #####
SRVAUTHS     := $(MODDIRS)/rpdutils.cxx $(MODDIRS)/ssh.cxx
SRVAUTHO     := $(call stripsrc,$(SRVAUTHS:.cxx=.o))

SRVAUTHLIB   := $(LPATH)/libSrvAuth.$(SOEXT)

##### Flags used in rootd amd proofd Module.mk #####
# use shadow passwords for authentication
ifneq ($(SHADOWFLAGS),)
SHADOWLIBS   := $(SHADOWLIBDIR) $(SHADOWLIB)
endif

# use AFS for authentication
ifneq ($(AFSLIB),)
AFSLIBS      := $(AFSLIBDIR) $(AFSLIB)
endif

# use SRP for authentication
ifneq ($(SRPLIB),)
SRPFLAGS     := $(SRPINCDIR:%=-I%)
SRPLIBS      := $(SRPLIBDIR) $(SRPLIB)
endif

# use krb5 for authentication
ifneq ($(KRB5LIB),)
KRB5FLAGS     := $(KRB5INCDIR:%=-I%)
KRB5LIBS      := $(KRB5LIBDIR) $(KRB5LIB)
endif

# use Globus for authentication
ifneq ($(GLOBUSLIB),)
GLBSFLAGS     := $(GLOBUSINCDIR:%=-I%)
GLBSLIBS      := $(GLOBUSLIBDIR) $(GLOBUSLIB)
SRVAUTHS      += $(MODDIRS)/globus.cxx
SRVAUTHO      += $(call stripsrc,$(MODDIRS)/globus.o)
else
GLBSFLAGS     := $(SSLINCDIR:%=-I%)
RPDUTILS      := $(filter-out $(MODDIRS)/globus.cxx,$(RPDUTILS))
RPDUTILO      := $(filter-out $(call stripsrc,$(MODDIRS)/globus.o),$(RPDUTILO))
GLBSLIBS      += $(SSLLIBDIR) $(SSLLIB)
endif

# Combined...
AUTHFLAGS     := $(EXTRA_AUTHFLAGS) $(SHADOWFLAGS) $(AFSFLAGS) $(SRPFLAGS) \
                 $(KRB5FLAGS) $(GLBSFLAGS)
AUTHLIBS      := $(SHADOWLIBS) $(AFSLIBS) \
                 $(SRPLIBS) $(KRB5LIBS) $(GLBSLIBS) \
                 $(COMERRLIBDIR) $(COMERRLIB) $(RESOLVLIB) \
                 $(CRYPTOLIBDIR) $(CRYPTOLIB)
ifeq ($(ARCH),win32gcc)
AUTHLIBS      += -lz
endif

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RPDUTILH))
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RPDCONNH))
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RPDPRIVH))
ALLLIBS       += $(SRVAUTHLIB)

# include all dependency files
INCLUDEFILES  += $(RPDUTILDEP) $(RPDCONNDEP) $(RPDPRIVDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(RPDUTILDIRI)/%.h
		cp $< $@

$(SRVAUTHLIB):  $(SRVAUTHO) $(RSAO) $(DAEMONUTILSO) $(STRLCPYO) $(ORDER_) $(MAINLIBS) $(SRVAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSrvAuth.$(SOEXT) $@ "$(SRVAUTHO) $(RSAO)" \
		   "$(DAEMONUTILSO) $(SRVAUTHLIBEXTRA) $(STRLCPYO) $(CRYPTLIBS) $(AUTHLIBS)"

all-$(MODNAME): $(RPDUTILO) $(RPDCONNO) $(RPDPRIVO) $(SRVAUTHLIB)

clean-$(MODNAME):
		@rm -f $(RPDUTILO) $(RPDCONNO) $(RPDPRIVO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RPDUTILDEP) $(RPDCONNDEP) $(RPDPRIVDEP) $(SRVAUTHLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(RPDUTILO): CXXFLAGS += $(AUTHFLAGS)
ifeq ($(MACOSX_SSL_DEPRECATED),yes)
$(call stripsrc,$(RPDUTILDIRS)/rpdutils.o): CXXFLAGS += -Wno-deprecated-declarations
endif
