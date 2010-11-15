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
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLEWH))
ALLLIBS     += $(GLEWLIB)

# include all dependency files
INCLUDEFILES += $(GLEWDEP)

ifeq ($(ARCH),win32)
GLLIBS       := opengl32.lib glu32.lib
endif
ifeq ($(MACOSX_MINOR),3)
GLEWLIBEXTRA += -lz
endif

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
		    $(GLEWLIBEXTRA) $(XLIBS) $(GLLIBS)"

all-$(MODNAME): $(GLEWLIB)

clean-$(MODNAME):
		@rm -f $(GLEWO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GLEWDEP) $(GLEWLIB)
		@rm -rf include/GL

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GLEWO):     CFLAGS += $(OPENGLINCDIR:%=-I%)
