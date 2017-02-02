# Module.mk for rootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := rootd
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTDDIR     := $(MODDIR)
ROOTDDIRS    := $(ROOTDDIR)/src
ROOTDDIRI    := $(ROOTDDIR)/inc

ifneq (,$(filter $(ARCH),win32gcc win64gcc))
AUTHLIBS      += -lz
endif

##### rootd #####
ROOTDH       := $(wildcard $(MODDIRI)/*.h)
ROOTDS       := $(wildcard $(MODDIRS)/*.cxx)
ROOTDO       := $(call stripsrc,$(ROOTDS:.cxx=.o))
ROOTDDEP     := $(ROOTDO:.o=.d)
ROOTD        := bin/rootd

# used in the main Makefile
ROOTDH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTDH))
ALLHDRS      += $(ROOTDH_REL)
ALLEXECS     += $(ROOTD)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(ROOTDH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += // link no-library-created \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(ROOTDDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ROOTDDIRI)/%.h
		cp $< $@

$(ROOTD):       $(ROOTDO) $(RSAO) $(SNPRINTFO) $(GLBPATCHO) $(RPDUTILO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTDO) $(RPDUTILO) $(GLBPATCHO) \
		   $(RSAO) $(SNPRINTFO) $(CRYPTLIBS) $(AUTHLIBS) $(STRLCPYO) $(SYSLIBS)

all-$(MODNAME): $(ROOTD)

clean-$(MODNAME):
		@rm -f $(ROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTDDEP) $(ROOTD)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(ROOTDO):  CXXFLAGS += $(AUTHFLAGS) -I$(ROOT_SRCDIR)/net/rpdutils/res
