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

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CHIRPH))
ALLLIBS     += $(CHIRPLIB)

# include all dependency files
INCLUDEFILES += $(CHIRPDEP)

##### local rules #####
include/%.h:    $(CHIRPDIRI)/%.h
		cp $< $@

$(CHIRPLIB):    $(CHIRPO) $(CHIRPDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libChirp.$(SOEXT) $@ "$(CHIRPO) $(CHIRPDO)" \
		   "$(CHIRPLIBEXTRA) $(CHIRPLIBDIR) $(CHIRPCLILIB)"

$(CHIRPDS):     $(CHIRPH) $(CHIRPL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CHIRPH) $(CHIRPL)

$(CHIRPDO):     $(CHIRPDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(CHIRPINCDIR:%=-I%) -I. -o $@ -c $<

all-chirp:      $(CHIRPLIB)

map-chirp:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(CHIRPLIB) \
		   -d $(CHIRPLIBDEP) -c $(CHIRPL)

map::           map-chirp

clean-chirp:
		@rm -f $(CHIRPO) $(CHIRPDO)

clean::         clean-chirp

distclean-chirp: clean-chirp
		@rm -f $(CHIRPDEP) $(CHIRPDS) $(CHIRPDH) $(CHIRPLIB)

distclean::     distclean-chirp

##### extra rules ######
$(CHIRPO): %.o: %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(CHIRPINCDIR:%=-I%) -o $@ -c $<
