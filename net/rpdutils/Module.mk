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
RPDUTILDIRR  := $(RPDUTILDIR)/res

##### $(RPDUTILO) #####
RPDUTILS     := $(filter-out $(MODDIRS)/rpdpriv.cxx, $(filter-out $(MODDIRS)/rpdconn.cxx, $(wildcard $(MODDIRS)/*.cxx)))
RPDUTILO     := $(call stripsrc,$(RPDUTILS:.cxx=.o))

RPDUTILDEP   := $(RPDUTILO:.o=.d)

##### $(RPDCONNO) #####
RPDCONNS     := $(MODDIRS)/rpdconn.cxx
RPDCONNO     := $(call stripsrc,$(RPDCONNS:.cxx=.o))

RPDCONNDEP   := $(RPDCONNO:.o=.d)

##### $(RPDPRIVO) #####
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
RPDALLO       := $(RPDUTILO) $(RPDCONNO) $(RPDPRIVO) $(SRVAUTHO)
AUTHFLAGS     := $(EXTRA_AUTHFLAGS) $(SHADOWFLAGS) $(AFSFLAGS) $(SRPFLAGS) \
                 $(KRB5FLAGS) $(GLBSFLAGS)
AUTHLIBS      := $(SHADOWLIBS) $(AFSLIBS) \
                 $(SRPLIBS) $(KRB5LIBS) $(GLBSLIBS) \
                 $(COMERRLIBDIR) $(COMERRLIB) $(RESOLVLIB) \
                 $(CRYPTOLIBDIR) $(CRYPTOLIB)
ifneq (,$(filter $(ARCH),win32gcc win64gcc))
AUTHLIBS      += -lz
endif

# used in the main Makefile
RPDH_REL      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RPDUTILH) $(RPDCONNH) $(RPDPRIVH))
ALLHDRS       += $(RPDH_REL)
ALLLIBS       += $(SRVAUTHLIB)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(RPDH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(SRVAUTHLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES  += $(RPDUTILDEP) $(RPDCONNDEP) $(RPDPRIVDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

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
$(RPDALLO):  CXXFLAGS += -I$(RPDUTILDIRR) -I$(AUTHDIRR)
$(RPDUTILO): CXXFLAGS += $(AUTHFLAGS)
ifeq ($(MACOSX_SSL_DEPRECATED),yes)
$(call stripsrc,$(RPDUTILDIRS)/rpdutils.o): CXXFLAGS += -Wno-deprecated-declarations
endif
