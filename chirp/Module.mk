# Module.mk for chirp module
#
# Author: Dan Bradley <dan@hep.wisc.edu>, 16/12/2002

MODDIR       := chirp
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CHIRPDIR    := $(MODDIR)
CHIRPDIRS   := $(CHIRPDIR)/src
CHIRPDIRI   := $(CHIRPDIR)/inc

##### libChirp #####
CHIRPL      := $(MODDIRI)/LinkDef.h
CHIRPDS     := $(MODDIRS)/G__Chirp.cxx
CHIRPDO     := $(CHIRPDS:.cxx=.o)
CHIRPDH     := $(CHIRPDS:.cxx=.h)

CHIRPH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CHIRPS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CHIRPO      := $(CHIRPS:.cxx=.o)

CHIRPDEP    := $(CHIRPO:.o=.d) $(CHIRPDO:.o=.d)

CHIRPLIB    := $(LPATH)/libChirp.$(SOEXT)
CHIRPMAP    := $(CHIRPLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CHIRPH))
ALLLIBS     += $(CHIRPLIB)
ALLMAPS     += $(CHIRPMAP)

# include all dependency files
INCLUDEFILES += $(CHIRPDEP)

##### local rules #####
include/%.h:    $(CHIRPDIRI)/%.h
		cp $< $@

$(CHIRPLIB):    $(CHIRPO) $(CHIRPDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libChirp.$(SOEXT) $@ "$(CHIRPO) $(CHIRPDO)" \
		   "$(CHIRPLIBEXTRA) $(CHIRPLIBDIR) $(CHIRPCLILIB)"

$(CHIRPDS):     $(CHIRPH) $(CHIRPL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CHIRPH) $(CHIRPL)

$(CHIRPMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(CHIRPL)
		$(RLIBMAP) -o $(CHIRPMAP) -l $(CHIRPLIB) \
		   -d $(CHIRPLIBDEPM) -c $(CHIRPL)

all-chirp:      $(CHIRPLIB) $(CHIRPMAP)

clean-chirp:
		@rm -f $(CHIRPO) $(CHIRPDO)

clean::         clean-chirp

distclean-chirp: clean-chirp
		@rm -f $(CHIRPDEP) $(CHIRPDS) $(CHIRPDH) $(CHIRPLIB) $(CHIRPMAP)

distclean::     distclean-chirp

##### extra rules ######
$(CHIRPO) $(CHIRPDO): CXXFLAGS += $(CHIRPINCDIR:%=-I%)
