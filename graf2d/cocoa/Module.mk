# Module.mk for cocoa module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Timur Pocheptsov, 22/11/2011

COCOANDEBUG := -DNDEBUG
ifeq ($(ROOTBUILD),debug)
   COCOANDEBUG :=
endif

MODNAME      := cocoa
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

COCOADIR     := $(MODDIR)
COCOADIRS    := $(COCOADIR)/src
COCOADIRI    := $(COCOADIR)/inc

##### libGCocoa #####
COCOAL       := $(MODDIRI)/LinkDef.h
COCOADS      := $(call stripsrc,$(MODDIRS)/G__Cocoa.cxx)
COCOADO      := $(COCOADS:.cxx=.o)
COCOADH      := $(COCOADS:.cxx=.h)

COCOAH1      := $(wildcard $(MODDIRI)/T*.h)
COCOAH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
COCOAS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
COCOAO       := $(call stripsrc,$(COCOAS:.cxx=.o))
COCOAOBJCPPS := $(wildcard $(MODDIRS)/*.mm)
COCOAOBJCPPO := $(call stripsrc,$(COCOAOBJCPPS:.mm=.o))

COCOADEP     := $(COCOAO:.o=.d) $(COCOADO:.o=.d) $(COCOAOBJCPPO:.o=.d)

COCOALIB     := $(LPATH)/libGCocoa.$(SOEXT)
COCOAMAP     := $(COCOALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
COCOAH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(COCOAH))
ALLHDRS      += $(COCOAH_REL)
ALLLIBS      += $(COCOALIB)
ALLMAPS      += $(COCOAMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(COCOAH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Graf2d_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += requires objc \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(COCOALIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(COCOADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(COCOADIRI)/%.h
		cp $< $@

$(COCOALIB):    $(COCOAO) $(COCOAOBJCPPO) $(COCOADO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) \
                $(COCOALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libGCocoa.$(SOEXT) $@ \
		   "$(COCOAO) $(COCOAOBJCPPO) $(COCOADO)" \
         "$(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(COCOALIBEXTRA) -framework Cocoa -framework OpenGL"

$(call pcmrule,COCOA)
	$(noop)

$(COCOADS):     $(COCOAH1) $(COCOAL) $(ROOTCLINGEXE) $(call pcmdep,COCOA)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,COCOA) -c $(FREETYPEINC) $(COCOAH1) $(COCOAL)

$(COCOAMAP):    $(COCOAH1) $(COCOAL) $(ROOTCLINGEXE) $(call pcmdep,COCOA)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(COCOADS) $(call dictModule,COCOA) -c $(FREETYPEINC) $(COCOAH1) $(COCOAL)

all-$(MODNAME): $(COCOALIB)

clean-$(MODNAME):
		@rm -f $(COCOAO) $(COCOADO) $(COCOAOBJCPPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(COCOADEP) $(COCOADS) $(COCOADH) $(COCOALIB) $(COCOAMAP)

distclean::     distclean-$(MODNAME)

ifeq ($(CXXMODULES),yes)
# We cannot compile objc/objc++ TU with -fmodules-local-submodule-visibility.
$(COCOAOBJCPPO) $(COCOADO) $(COCOAO):
OBJCXXFLAGS := $(CXXFLAGS) $(COCOANDEBUG) $(FREETYPEINC)
# FIXME: Until we resolve the TMVA Pattern vs cocoa Pattern Quick*.h
OBJCXXFLAGS  := $(filter-out $(ROOT_CXXMODULES_CXXFLAGS),$(OBJCXXFLAGS))
else
$(COCOAOBJCPPO) $(COCOADO) $(COCOAO): OBJCXXFLAGS := $(CXXFLAGS) $(COCOANDEBUG) $(FREETYPEINC)
endif
