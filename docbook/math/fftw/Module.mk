# Module.mk for fftw module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 23/1/2006

MODNAME      := fftw
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FFTWDIR      := $(MODDIR)
FFTWDIRS     := $(FFTWDIR)/src
FFTWDIRI     := $(FFTWDIR)/inc

#### libFFTW ####
FFTWL        := $(MODDIRI)/LinkDef.h
FFTWDS       := $(call stripsrc,$(MODDIRS)/G__FFTW.cxx)
FFTWDO       := $(FFTWDS:.cxx=.o)
FFTWDH       := $(FFTWDS:.cxx=.h)

FFTWH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FFTWS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FFTWO        := $(call stripsrc,$(FFTWS:.cxx=.o))

FFTWDEP      := $(FFTWO:.o=.d) $(FFTWDO:.o=.d)

FFTWLIB      := $(LPATH)/libFFTW.$(SOEXT)
FFTWMAP      := $(FFTWLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FFTWH))
ALLLIBS      += $(FFTWLIB)
ALLMAPS      += $(FFTWMAP)

# include all dependency files
INCLUDEFILES += $(FFTWDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FFTWDIRI)/%.h
		cp $< $@

$(FFTWLIB):     $(FFTWO) $(FFTWDO) $(ORDER_) $(MAINLIBS) $(FFTWLIBDEP)
	       	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFFTW.$(SOEXT) $@ "$(FFTWO) $(FFTWDO)" \
		   "$(FFTWLIBEXTRA) $(FFTW3LIBDIR) $(FFTW3LIB)"

$(FFTWDS):      $(FFTWH) $(FFTWL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FFTWH) $(FFTWL)

$(FFTWMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(FFTWL)
		$(RLIBMAP) -o $@ -l $(FFTWLIB) \
		   -d $(FFTWLIBDEPM) -c $(FFTWL)

all-$(MODNAME): $(FFTWLIB) $(FFTWMAP)

clean-$(MODNAME):
		@rm -f $(FFTWO) $(FFTWDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FFTWDEP) $(FFTWDS) $(FFTWDH) $(FFTWLIB) $(FFTWMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(FFTWO) $(FFTWDO): CXXFLAGS += $(FFTW3INCDIR:%=-I%)
