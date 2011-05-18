# Module.mk for chirp module
#
# Author: Dan Bradley <dan@hep.wisc.edu>, 16/12/2002

MODNAME     := chirp
MODDIR      := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS     := $(MODDIR)/src
MODDIRI     := $(MODDIR)/inc

CHIRPDIR    := $(MODDIR)
CHIRPDIRS   := $(CHIRPDIR)/src
CHIRPDIRI   := $(CHIRPDIR)/inc

##### libChirp #####
CHIRPL      := $(MODDIRI)/LinkDef.h
CHIRPDS     := $(call stripsrc,$(MODDIRS)/G__Chirp.cxx)
CHIRPDO     := $(CHIRPDS:.cxx=.o)
CHIRPDH     := $(CHIRPDS:.cxx=.h)

CHIRPH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CHIRPS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CHIRPO      := $(call stripsrc,$(CHIRPS:.cxx=.o))

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
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CHIRPDIRI)/%.h
		cp $< $@

$(CHIRPLIB):    $(CHIRPO) $(CHIRPDO) $(ORDER_) $(MAINLIBS) $(CHIRPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libChirp.$(SOEXT) $@ "$(CHIRPO) $(CHIRPDO)" \
		   "$(CHIRPLIBEXTRA) $(CHIRPLIBDIR) $(CHIRPCLILIB)"

$(CHIRPDS):     $(CHIRPH) $(CHIRPL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CHIRPH) $(CHIRPL)

$(CHIRPMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(CHIRPL)
		$(RLIBMAP) -o $@ -l $(CHIRPLIB) \
		   -d $(CHIRPLIBDEPM) -c $(CHIRPL)

all-$(MODNAME): $(CHIRPLIB) $(CHIRPMAP)

clean-$(MODNAME):
		@rm -f $(CHIRPO) $(CHIRPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CHIRPDEP) $(CHIRPDS) $(CHIRPDH) $(CHIRPLIB) $(CHIRPMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(CHIRPO) $(CHIRPDO): CXXFLAGS += $(CHIRPINCDIR:%=-I%)
