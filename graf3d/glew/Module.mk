# Module.mk for glew module
# Copyright (c) 2009 Rene Brun and Fons Rademakers
#
# Author: Matevz & Timur, 8/5/2009

MODNAME      := glew
MODDIR       := $(ROOT_SRCDIR)/graf3d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLEWDIR      := $(MODDIR)
GLEWDIRS     := $(GLEWDIR)/src
GLEWDIRI     := $(GLEWDIR)/inc

##### libGLEW #####
GLEWH        := $(filter-out $(MODDIRI)/GL/LinkDef%,$(wildcard $(MODDIRI)/GL/*.h))
GLEWS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.c))
GLEWO        := $(call stripsrc,$(GLEWS:.c=.o))

GLEWDEP      := $(GLEWO:.o=.d)

GLEWLIB      := $(LPATH)/libGLEW.$(SOEXT)

# used in the main Makefile
GLEWH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLEWH))
ALLHDRS     += $(GLEWH_REL)
ALLLIBS     += $(GLEWLIB)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,exclude header \"%\"\\n,$(GLEWH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Graph3d_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(GLEWLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(GLEWDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/GL/%.h: $(GLEWDIRI)/GL/%.h
		@(if [ ! -d "include/GL" ]; then     \
		   mkdir -p include/GL;              \
		fi)
		cp $< $@

$(GLEWLIB):     $(GLEWO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) $(GLEWLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGLEW.$(SOEXT) $@ \
		   "$(GLEWO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(GLEWLIBEXTRA) $(GLLIBS)"

all-$(MODNAME): $(GLEWLIB)

clean-$(MODNAME):
		@rm -f $(GLEWO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GLEWDEP) $(GLEWLIB)
		@rm -rf include/GL

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GLEWO): CFLAGS += $(OPENGLINCDIR:%=-I%)
