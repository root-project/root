# Module.mk for gdml module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ben Lloyd 09/11/06

MODDIR       := gdml
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GDMLDIR      := $(MODDIR)
GDMLDIRS     := $(GDMLDIR)/src
GDMLDIRI     := $(GDMLDIR)/inc

##### libGdml #####
GDMLL        := $(MODDIRI)/LinkDef.h
GDMLDS       := $(MODDIRS)/G__Gdml.cxx
GDMLDO       := $(GDMLDS:.cxx=.o)
GDMLDH       := $(GDMLDS:.cxx=.h)

GDMLH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GDMLS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GDMLO        := $(GDMLS:.cxx=.o)

GDMLDEP      := $(GDMLO:.o=.d) $(GDMLDO:.o=.d)

GDMLLIB      := $(LPATH)/libGdml.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GDMLH))
ALLLIBS      += $(GDMLLIB)

# include all dependency files
INCLUDEFILES += $(GDMLDEP)

# These are undefined if using an external XROOTD distribution
# The new XROOTD build system based on autotools installs the headers
# under <dir>/include/xrootd, while the old system under <dir>/src
ifneq ($(XROOTDDIR),)
ifeq ($(XROOTDDIRI),)
XROOTDDIRI   := $(XROOTDDIR)/include/xrootd
ifeq ($(wildcard $(XROOTDDIRI)/*.hh),)
XROOTDDIRI   := $(XROOTDDIR)/src
endif
XROOTDDIRL   := $(XROOTDDIR)/lib
endif
endif

# Xrootd includes
GDMLINCEXTRA := $(XROOTDDIRI:%=-I%)

# Xrootd client libs
ifeq ($(PLATFORM),win32)
GDMLLIBEXTRA += $(XROOTDDIRL)/libXrdClient.lib
else
GDMLLIBEXTRA += $(XROOTDDIRL)/libXrdClient.a $(XROOTDDIRL)/libXrdOuc.a \
		$(XROOTDDIRL)/libXrdNet.a
endif

##### local rules #####
include/%.h:    $(GDMLDIRI)/%.h
		cp $< $@

$(GDMLLIB):     $(GDMLO) $(GDMLDO) $(XRDPLUGINS) $(ORDER_) $(MAINLIBS) $(GDMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGdml.$(SOEXT) $@ "$(GDMLO) $(GDMLDO)" \
		   "$(GDMLLIBEXTRA)"

$(GDMLDS):      $(GDMLH1) $(GDMLL) $(ROOTCINTTMPEXE) $(XROOTDETAG)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GDMLINCEXTRA) $(GDMLH) $(GDMLL)

all-gdml:       $(GDMLLIB)

map-gdml:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GDMLLIB) \
		   -d $(GDMLLIBDEP) -c $(GDMLL)

map::           map-gdml

clean-gdml:
		@rm -f $(GDMLO) $(GDMLDO)

clean::         clean-gdml

distclean-gdml: clean-gdml
		@rm -f $(GDMLDEP) $(GDMLDS) $(GDMLDH) $(GDMLLIB)

distclean::     distclean-gdml

##### extra rules ######
$(GDMLO) $(GDMLDO): $(XROOTDETAG)
$(GDMLO) $(GDMLDO): CXXFLAGS += $(GDMLINCEXTRA)
