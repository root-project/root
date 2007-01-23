# Module.mk for auth module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 7/2005

MODDIR       := auth
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

AUTHDIR      := $(MODDIR)
AUTHDIRS     := $(AUTHDIR)/src
AUTHDIRI     := $(AUTHDIR)/inc

##### libRootAuth #####
RAUTHL       := $(MODDIRI)/LinkDefRoot.h
RAUTHDS      := $(MODDIRS)/G__RootAuth.cxx
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

RAUTHO       := $(RAUTHS:.cxx=.o)

#### for libSrvAuth (built in rpdutils/Module.mk) ####
DAEMONUTILSO := $(MODDIRS)/DaemonUtils.o

RAUTHDEP     := $(RAUTHO:.o=.d) $(RAUTHDO:.o=.d) $(DAEMONUTILSO:.o=.d)

RAUTHLIB     := $(LPATH)/libRootAuth.$(SOEXT)

##### libAFSAuth #####
ifneq ($(AFSLIB),)
AFSAUTHL       := $(MODDIRI)/LinkDefAFS.h
AFSAUTHDS      := $(MODDIRS)/G__AFSAuth.cxx
AFSAUTHDO      := $(AFSAUTHDS:.cxx=.o)
AFSAUTHDH      := $(AFSAUTHDS:.cxx=.h)

AFSAUTHH     := $(MODDIRI)/AFSAuth.h $(MODDIRI)/AFSAuthTypes.h $(MODDIRI)/TAFS.h
AFSAUTHS     := $(MODDIRS)/AFSAuth.cxx $(MODDIRS)/TAFS.cxx

AFSAUTHO     := $(AFSAUTHS:.cxx=.o)

AFSAUTHDEP   := $(AFSAUTHO:.o=.d) $(AFSAUTHDO:.o=.d)

AFSAUTHLIB   := $(LPATH)/libAFSAuth.$(SOEXT)
endif

#### for libSrvAuth (built in rpdutils/Module.mk) ####
DAEMONUTILSO := $(MODDIRS)/DaemonUtils.o

#### for rootd and proofd ####
RSAO         := $(AUTHDIRS)/rsaaux.o $(AUTHDIRS)/rsalib.o $(AUTHDIRS)/rsafun.o
ifneq ($(AFSLIB),)
RSAO         += $(MODDIRS)/AFSAuth.o
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
ifneq ($(AFSLIB),)
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(AFSAUTHH))
ALLLIBS      += $(AFSAUTHLIB)
endif

# include all dependency files
INCLUDEFILES += $(RAUTHDEP)
ifneq ($(AFSLIB),)
INCLUDEFILES += $(AFSAUTHDEP)
endif

##### local rules #####
include/%.h:    $(AUTHDIRI)/%.h
		cp $< $@

$(RAUTHLIB):    $(RAUTHO) $(RAUTHDO) $(ORDER_) $(MAINLIBS) $(RAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRootAuth.$(SOEXT) $@ "$(RAUTHO) $(RAUTHDO)" \
		   "$(RAUTHLIBEXTRA) $(EXTRA_RAUTHLIBS)"

$(RAUTHDS):     $(RAUTHH) $(RAUTHL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RAUTHH) $(RAUTHL)

$(AFSAUTHLIB):  $(AFSAUTHO) $(AFSAUTHDO) $(ORDER_) $(MAINLIBS) $(AFSAUTHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libAFSAuth.$(SOEXT) $@ "$(AFSAUTHO) $(AFSAUTHDO)" \
		   "$(AFSLIBDIR) $(AFSLIB) $(RESOLVLIB)"

$(AFSAUTHDS):   $(AFSAUTHH) $(AFSAUTHL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(AFSAUTHH) $(AFSAUTHL)

all-auth:       $(RAUTHLIB) $(AFSAUTHLIB)

map-auth:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(RAUTHLIB) \
		   -d $(RAUTHLIBDEP) -c $(RAUTHL)

map-afs:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(AFSAUTHLIB) \
		   -d $(AFSAUTHLIBDEP) -c $(AFSAUTHL)

map::           map-auth map-afs

clean-auth:
		@rm -f $(RAUTHO) $(RAUTHDO) $(DAEMONUTILSO) $(AFSAUTHO) $(AFSAUTHDO)

clean::         clean-auth

distclean-auth: clean-auth
		@rm -f $(RAUTHDEP) $(RAUTHDS) $(RAUTHDH) $(RAUTHLIB)\
		       $(AFSAUTHDEP) $(AFSAUTHDS) $(AFSAUTHLIB)

distclean::     distclean-auth

##### extra rules ######
$(RAUTHO):      CXXFLAGS += $(EXTRA_RAUTHFLAGS)
$(AFSAUTHO):    CXXFLAGS += $(AFSINCDIR)
