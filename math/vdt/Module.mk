# Module.mk for VDT
#
# Author: Danilo Piparo

MODNAME       := vdt
MODDIR        := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRI       := $(MODDIR)/include
MODDIRS       := $(MODDIR)/include

VDTH    := include/vdt/asin.h\
	include/vdt/atan2.h\
	include/vdt/atan.h\
	include/vdt/cos.h\
	include/vdt/exp.h\
	include/vdt/inv.h\
	include/vdt/log.h\
	include/vdt/sincos.h\
	include/vdt/sin.h\
	include/vdt/sqrt.h\
	include/vdt/tan.h\
	include/vdt/vdtcore_common.h\
	include/vdt/vdtMath.h

# We create a stamp file as vdt is the only module without a library.
VDTH_REL     := $(patsubst $(MODDIRI)/vdt/%.h, include/%.h, $(VDTH))
ALLHDRS      += $(VDTH_REL)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(VDTH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += // link no-library-created \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/vdt/%.h: $(MODDIRI)/vdt/%.h
		@(if [ ! -d "include/vdt" ]; then   \
		mkdir -p include/vdt;       \
		fi)
		cp $< $@

all-$(MODNAME): 
	$(noop)

clean-$(MODNAME):
	$(noop)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -rf include/vdt

distclean:: distclean-$(MODNAME)
