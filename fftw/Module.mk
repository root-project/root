# Module.mk for fft module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 23/1/2006

MODDIR       := fftw
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FFTWDIR       := $(MODDIR)
FFTWDIRS      := $(FFTWDIR)/src
FFTWDIRI      := $(FFTWDIR)/inc


#### libFFTW ####
FFTWL     := $(MODDIRI)/LinkDef.h
FFTWDS    := $(MODDIRS)/G__FFTW.cxx
FFTWDO    := $(FFTWDS:.cxx=.o)
FFTWDH    := $(FFTWDS:.cxx=.h)

FFTWH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FFTWS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FFTWO     := $(FFTWS:.cxx=.o)

FFTWDEP   := $(FFTWO:.o=.d) $(FFTWDO:.o=.d)

FFTWLIB   := $(LPATH)/libFFTW.$(SOEXT)

# used in the main Makefile
ALLHDRS         += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FFTWH))
ALLLIBS         += $(FFTWLIB)

# include all dependency files
INCLUDEFILES    += $(FFTWDEP)

##### local rules #####
include/%.h:    $(FFTWDIRI)/%.h
		cp $< $@



$(FFTWLIB):     $(FFTWO) $(FFTWDO) $(ORDER_) \
                  $(MAINLIBS) $(FFTWLIBDEP)
	       	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFFTW.$(SOEXT) $@  \
		   "$(FFTWO) $(FFTWDO)"        \
		   "$(FFTWLIBEXTRA) $(FFTW3LIBDIR) $(FFTW3LIB)"

$(FFTWDS):      $(FFTWH) $(FFTWL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FFTWH) $(FFTWL)

#$(FFTWDO): $(FFTWDS) $(FFTWLIBA)
#	$(CXX) $(NOOPT) $(CXXFLAGS) $(FFTWDIRI) -I. -o $@ -c $<

all-fft:        $(FFTWLIB)

map-fft:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(FFTWLIB) \
		   -d $(FFTWLIBDEP) -c $(FFTWL)

map::           map-fft

clean-fft:
		@rm -f $(FFTWO) $(FFTWDO)

clean::         clean-fft

distclean-fft:  clean-fft
		@rm -f $(FFTWDEP) $(FFTWDS) $(FFTWDH) \
		   $(FFTWLIB)
		@rm -rf $(FFTWDIRS)

distclean::     distclean-fft

##### extra rules ######
$(FFTWO) $(FFTWDO): CXXFLAGS += -I$(FFTW3INCDIR)
