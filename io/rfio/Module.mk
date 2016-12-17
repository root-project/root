# Module.mk for rfio module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := rfio
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RFIODIR      := $(MODDIR)
RFIODIRS     := $(RFIODIR)/src
RFIODIRI     := $(RFIODIR)/inc

##### libRFIO #####
RFIOL        := $(MODDIRI)/LinkDef.h
RFIODS       := $(call stripsrc,$(MODDIRS)/G__RFIO.cxx)
RFIODO       := $(RFIODS:.cxx=.o)
RFIODH       := $(RFIODS:.cxx=.h)

RFIOH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
RFIOS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RFIOO        := $(call stripsrc,$(RFIOS:.cxx=.o))

RFIODEP      := $(RFIOO:.o=.d) $(RFIODO:.o=.d)

RFIOLIB      := $(LPATH)/libRFIO.$(SOEXT)
RFIOMAP      := $(RFIOLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
RFIOH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(RFIOH))
ALLHDRS     += $(RFIOH_REL)
ALLLIBS     += $(RFIOLIB)
ALLMAPS     += $(RFIOMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(RFIOH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Io_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(RFIOLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(RFIODEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(RFIODIRI)/%.h
		cp $< $@

$(RFIOLIB):     $(RFIOO) $(RFIODO) $(ORDER_) $(MAINLIBS) $(RFIOLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRFIO.$(SOEXT) $@ "$(RFIOO) $(RFIODO)" \
		   "$(SHIFTLIBDIR) $(SHIFTLIB) $(RFIOLIBEXTRA)"

$(call pcmrule,RFIO)
	$(noop)

$(RFIODS):      $(RFIOH) $(RFIOL) $(ROOTCLINGEXE) $(call pcmdep,RFIO)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,RFIO) -c $(RFIOH) $(RFIOL)

$(RFIOMAP):     $(RFIOH) $(RFIOL) $(ROOTCLINGEXE) $(call pcmdep,RFIO)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(RFIODS) $(call dictModule,RFIO) -c $(RFIOH) $(RFIOL)

all-$(MODNAME): $(RFIOLIB)

clean-$(MODNAME):
		@rm -f $(RFIOO) $(RFIODO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RFIODEP) $(RFIODS) $(RFIODH) $(RFIOLIB) $(RFIOMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(RFIOO) $(RFIODO): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))

ifeq ($(PLATFORM),win32)
$(RFIOO): CXXFLAGS += $(SHIFTCFLAGS) $(SHIFTINCDIR:%=-I%) -DNOGDI -D__INSIDE_CYGWIN__
ifneq (yes,$(WINRTDEBUG))
$(RFIOLIB): LDFLAGS += -nodefaultlib:msvcrtd.lib 
endif
else
$(RFIOO): CXXFLAGS += $(SHIFTCFLAGS) $(SHIFTINCDIR:%=-I%)
endif
