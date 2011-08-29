# Module.mk for auth module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 7/2005

MODNAME      := auth
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

AUTHDIR      := $(MODDIR)
AUTHDIRS     := $(AUTHDIR)/src
AUTHDIRI     := $(AUTHDIR)/inc

##### libRootAuth #####
RAUTHL       := $(MODDIRI)/LinkDefRoot.h
RAUTHDS      := $(call stripsrc,$(MODDIRS)/G__RootAuth.cxx)
RAUTHDO      := $(RAUTHDS:.cxx=.o)
RAUTHDH      := $(RAUTHDS:.cxx=.h)

RAUTHH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RAUTHS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

RAUTHH       := $(filter-out $(MODDIRI)/DaemonUtils.h,$(RAUTHH))
RAUTHS       := $(filter-out $(MODDIRS)/DaemonUtils.cxx,$(RAUTHS))
RAUTHH       := $(filter-out $(MODDIRI)/AFSAuth.h,$(RAUTHH))
RAUTHH       := $(filter-out $(MODDIRI)/AFSAuthTypes.h,$(RAUTHH))
RAUTHS       := $(filter-out $(MODDIRS)/AFSAuth.cxx,$(RAUTHS))
RAUTHH       := $(filter-out $(MODDIRI)/TAFS.h,$(RAUTHH))
RAUTHS       := $(filter-out $(MODDIRS)/TAFS.cxx,$(RAUTHS))

RAUTHO       := $(call stripsrc,$(RAUTHS:.cxx=.o))

#### for libSrvAuth (built in rpdutils/Module.mk) ####
DAEMONUTILSO := $(call stripsrc,$(MODDIRS)/DaemonUtils.o)

RAUTHDEP     := $(RAUTHO:.o=.d) $(RAUTHDO:.o=.d) $(DAEMONUTILSO:.o=.d)

RAUTHLIB     := $(LPATH)/libRootAuth.$(SOEXT)
RAUTHMAP     := $(RAUTHLIB:.$(SOEXT)=.rootmap)

##### libAFSAuth #####
ifneq ($(AFSLIB),)
AFSAUTHL     := $(MODDIRI)/LinkDefAFS.h
AFSAUTHDS    := $(call stripsrc,$(MODDIRS)/G__AFSAuth.cxx)
AFSAUTHDO    := $(AFSAUTHDS:.cxx=.o)
AFSAUTHDH    := $(AFSAUTHDS:.cxx=.h)

AFSAUTHH     := $(MODDIRI)/AFSAuth.h $(MODDIRI)/AFSAuthTypes.h $(MODDIRI)/TAFS.h
AFSAUTHS     := $(MODDIRS)/AFSAuth.cxx $(MODDIRS)/TAFS.cxx

AFSAUTHO     := $(call stripsrc,$(AFSAUTHS:.cxx=.o))

AFSAUTHDEP   := $(AFSAUTHO:.o=.d) $(AFSAUTHDO:.o=.d)

AFSAUTHLIB   := $(LPATH)/libAFSAuth.$(SOEXT)
AFSAUTHMAP   := $(AFSAUTHLIB:.$(SOEXT)=.rootmap)
endif

#### for rootd and proofd ####
RSAO         := $(call stripsrc,$(AUTHDIRS)/rsaaux.o $(AUTHDIRS)/rsalib.o $(AUTHDIRS)/rsafun.o)
ifneq ($(AFSLIB),)
RSAO         += $(call stripsrc,$(MODDIRS)/AFSAuth.o)
endif

# Add SSL flags, if required
EXTRA_RAUTHFLAGS = $(EXTRA_AUTHFLAGS)
EXTRA_RAUTHLIBS  = $(CRYPTLIBS)
ifneq ($(SSLLIB),)
EXTRA_RAUTHFLAGS += $(SSLINCDIR:%=-I%)
EXTRA_RAUTHLIBS  += $(SSLLIBDIR) $(SSLLIB)
endif

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(RAUTHH)) \
                include/DaemonUtils.h
ALLLIBS      += $(RAUTHLIB)
ALLMAPS      += $(RAUTHMAP)
ifneq ($(AFSLIB),)
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(AFSAUTHH))
ALLLIBS      += $(AFSAUTHLIB)
ALLMAPS      += $(AFSAUTHMAP)
endif

# include all dependency files
INCLUDEFILES += $(RAUTHDEP)
ifneq ($(AFSLIB),)
INCLUDEFILES += $(AFSAUTHDEP)
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(AUTHDIRI)/%.h
		cp $< $@

$(RAUTHLIB):    $(RAUTHO) $(RAUTHDO) $(ORDER_) $(MAINLIBS) $(RAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRootAuth.$(SOEXT) $@ "$(RAUTHO) $(RAUTHDO)" \
		   "$(RAUTHLIBEXTRA) $(EXTRA_RAUTHLIBS)"

$(RAUTHDS):     $(RAUTHH) $(RAUTHL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RAUTHH) $(RAUTHL)

$(RAUTHMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(RAUTHL)
		$(RLIBMAP) -o $@ -l $(RAUTHLIB) \
		   -d $(RAUTHLIBDEPM) -c $(RAUTHL)

$(AFSAUTHLIB):  $(AFSAUTHO) $(AFSAUTHDO) $(ORDER_) $(MAINLIBS) $(AFSAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libAFSAuth.$(SOEXT) $@ \
		   "$(AFSAUTHO) $(AFSAUTHDO)" \
		   "$(AFSLIBDIR) $(AFSLIB) $(RESOLVLIB)"

$(AFSAUTHDS):   $(AFSAUTHH) $(AFSAUTHL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(AFSAUTHH) $(AFSAUTHL)

$(AFSAUTHMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(AFSAUTHL)
		$(RLIBMAP) -o $(AFSAUTHMAP) -l $(AFSAUTHLIB) \
		   -d $(AFSAUTHLIBDEPM) -c $(AFSAUTHL)

all-$(MODNAME): $(RAUTHLIB) $(AFSAUTHLIB) $(RAUTHMAP) $(AFSAUTHMAP)

clean-$(MODNAME):
		@rm -f $(RAUTHO) $(RAUTHDO) $(DAEMONUTILSO) $(AFSAUTHO) \
		       $(AFSAUTHDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RAUTHDEP) $(RAUTHDS) $(RAUTHDH) $(RAUTHLIB) \
		       $(AFSAUTHDEP) $(AFSAUTHDS) $(AFSAUTHLIB) \
		       $(RAUTHMAP) $(AFSAUTHMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(RAUTHO):      CXXFLAGS += $(EXTRA_RAUTHFLAGS)
$(AFSAUTHO):    CXXFLAGS += $(AFSINCDIR) $(AFSEXTRACFLAGS)
ifeq ($(MACOSX_MINOR),7)
$(call stripsrc,$(AUTHDIRS)/TAuthenticate.o): CXXFLAGS += -Wno-deprecated-declarations
endif
