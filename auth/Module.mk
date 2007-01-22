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

RAUTHO       := $(RAUTHS:.cxx=.o)

#### for libSrvAuth (built in rpdutils/Module.mk) ####
DAEMONUTILSO := $(MODDIRS)/DaemonUtils.o

RAUTHDEP     := $(RAUTHO:.o=.d) $(RAUTHDO:.o=.d) $(DAEMONUTILSO:.o=.d)

RAUTHLIB     := $(LPATH)/libRootAuth.$(SOEXT)

#### for rootd and proofd ####
RSAO         := $(AUTHDIRS)/rsaaux.o $(AUTHDIRS)/rsalib.o $(AUTHDIRS)/rsafun.o

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

# include all dependency files
INCLUDEFILES += $(RAUTHDEP)

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

all-auth:       $(RAUTHLIB)

map-auth:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(RAUTHLIB) \
		   -d $(RAUTHLIBDEP) -c $(RAUTHL)

map::           map-auth

clean-auth:
		@rm -f $(RAUTHO) $(RAUTHDO) $(DAEMONUTILSO)

clean::         clean-auth

distclean-auth: clean-auth
		@rm -f $(RAUTHDEP) $(RAUTHDS) $(RAUTHDH) $(RAUTHLIB)

distclean::     distclean-auth

##### extra rules ######
$(RAUTHO):      CXXFLAGS += $(EXTRA_RAUTHFLAGS)
