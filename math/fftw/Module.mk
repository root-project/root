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
FFTWH_REL    := $(patsubst $(MODDIRI)/%.h,include/%.h,$(FFTWH))
ALLHDRS      += $(FFTWH_REL)
ALLLIBS      += $(FFTWLIB)
ALLMAPS      += $(FFTWMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(FFTWH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_FFTW { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(FFTWLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

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

$(call pcmrule,FFTW)
	$(noop)

$(FFTWDS):      $(FFTWH) $(FFTWL) $(ROOTCLINGEXE) $(call pcmdep,FFTW)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,FFTW) -c $(FFTWH) $(FFTWL)

$(FFTWMAP):     $(FFTWH) $(FFTWL) $(ROOTCLINGEXE) $(call pcmdep,FFTW)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(FFTWDS) $(call dictModule,FFTW) -c $(FFTWH) $(FFTWL)

all-$(MODNAME): $(FFTWLIB)

clean-$(MODNAME):
		@rm -f $(FFTWO) $(FFTWDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FFTWDEP) $(FFTWDS) $(FFTWDH) $(FFTWLIB) $(FFTWMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(FFTWO) $(FFTWDO): CXXFLAGS += $(FFTW3INCDIR:%=-I%)
