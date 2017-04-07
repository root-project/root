# Module.mk for pyroot module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Pere Mato, Wim Lavrijsen, 22/4/2004

MODNAME      := pyroot
MODDIR       := $(ROOT_SRCDIR)/bindings/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PYROOTDIR    := $(MODDIR)
PYROOTDIRS   := $(PYROOTDIR)/src
PYROOTDIRI   := $(PYROOTDIR)/inc

##### libPyROOT #####
PYROOTL      := $(MODDIRI)/LinkDef.h
PYROOTDS     := $(call stripsrc,$(MODDIRS)/G__PyROOT.cxx)
PYROOTDO     := $(PYROOTDS:.cxx=.o)
PYROOTDH     := $(PYROOTDS:.cxx=.h)

PYROOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PYROOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PYROOTO      := $(call stripsrc,$(PYROOTS:.cxx=.o))

PYROOTDEP    := $(PYROOTO:.o=.d) $(PYROOTDO:.o=.d)

PYROOTLIB    := $(LPATH)/libPyROOT.$(SOEXT)
ifeq ($(ARCH),win32)
PYROOTPYD    := bin/$(notdir $(PYROOTLIB:.$(SOEXT)=.pyd))
endif
PYROOTMAP    := $(PYROOTLIB:.$(SOEXT)=.rootmap)

ROOTPYS      := $(wildcard $(MODDIR)/*.py)
ROOTAASS     := $(wildcard $(MODDIR)/ROOTaaS/* $(MODDIR)/ROOTaaS/*/* $(MODDIR)/ROOTaaS/*/*/*)
# Above includes ROOTaaS/config which is a directory; filter those out.
# Problem: $(dir $(ROOTAASS)) gives ROOTaaS/config/ thus patsubst %/, %
ROOTAASS     := $(filter-out $(sort $(patsubst %/,%,$(dir $(ROOTAASS)))),$(ROOTAASS))

ifeq ($(ARCH),win32)
ROOTPY       := $(subst $(MODDIR),bin,$(ROOTPYS))
ROOTAAS      := $(subst $(MODDIR),bin,$(ROOTAASS))
bin/%.py: $(MODDIR)/%.py
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp $< $@
bin/ROOTaaS/%: $(MODDIR)/ROOTaaS/%
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp -R $< $@
else
ROOTPY       := $(subst $(MODDIR),$(LPATH),$(ROOTPYS))
ROOTAAS      := $(subst $(MODDIR),$(LPATH),$(ROOTAASS))
$(LPATH)/%.py: $(MODDIR)/%.py
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp $< $@
$(LPATH)/ROOTaaS/%: $(MODDIR)/ROOTaaS/%
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cp -R $< $@
endif
ROOTPYC      := $(ROOTPY:.py=.pyc)
ROOTPYO      := $(ROOTPY:.py=.pyo)
ROOTAASC     := $(ROOTAAS:.py=.pyc)
ROOTAASO     := $(ROOTAAS:.py=.pyo)

# used in the main Makefile
PYROOTH_REL  := $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYROOTH))
ALLHDRS      += $(PYROOTH_REL)
ALLLIBS      += $(PYROOTLIB)
ALLMAPS      += $(PYROOTMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(PYROOTH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module $(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(PYROOTLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(PYROOTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PYROOTDIRI)/%.h
		cp $< $@

%.pyc: %.py;    python -c 'import py_compile; py_compile.compile( "$<" )'
%.pyo: %.py;    python -O -c 'import py_compile; py_compile.compile( "$<" )'

$(PYROOTLIB):   $(PYROOTO) $(PYROOTDO) $(ROOTPY) $(ROOTPYC) $(ROOTPYO) \
                $(ROOTLIBSDEP) $(PYTHONLIBDEP) \
                $(ROOTAAS) $(ROOTAASC) $(ROOTAASO)

		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		  "$(SOFLAGS)" libPyROOT.$(SOEXT) $@ \
		  "$(PYROOTO) $(PYROOTDO)" \
		  "$(ROOTULIBS) $(RPATH) $(ROOTLIBS) $(PYROOTLIBEXTRA) \
		   $(PYTHONLIBDIR) $(PYTHONLIB) $(PYTHONLIBFLAGS)"
ifeq ($(ARCH),win32)
	link -dll -nologo -IGNORE:4001 -machine:ix86 -export:initlibPyROOT \
	lib/libPyROOT.lib -nodefaultlib kernel32.lib msvcrt.lib \
	-out:$(PYROOTPYD)
	@(if [ -f $(PYROOTPYD).manifest ]; then \
		mt -nologo -manifest $(PYROOTPYD).manifest \
			-outputresource\:$(PYROOTPYD)\;2 ; \
		rm -f $(PYROOTPYD).manifest ; \
	fi)
	@rm -f bin/libPyROOT.lib
	@rm -f bin/libPyROOT.exp
endif

$(call pcmrule,PYROOT)
	$(noop)

$(PYROOTDS):    $(PYROOTH) $(PYROOTL) $(ROOTCLINGEXE) $(call pcmdep,PYROOT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PYROOT) -c -writeEmptyRootPCM $(PYROOTH) $(PYROOTL)

$(PYROOTMAP):   $(PYROOTH) $(PYROOTL) $(ROOTCLINGEXE) $(call pcmdep,PYROOT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(PYROOTDS) $(call dictModule,PYROOT) -c $(PYROOTH) $(PYROOTL)

all-$(MODNAME): $(PYROOTLIB)

clean-$(MODNAME):
		@rm -f $(PYROOTO) $(PYROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PYROOTDEP) $(PYROOTDS) $(PYROOTDH) $(PYROOTLIB) \
		   $(ROOTPY) $(ROOTPYC) $(ROOTPYO) $(PYROOTMAP) \
		   $(PYROOTPYD)
		@rm -rf $(LPATH)/ROOTaaS bin/ROOTaaS

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PYROOTO): CXXFLAGS += $(PYTHONINCDIR:%=-I%)
ifeq ($(GCC_MAJOR),4)
$(PYROOTO): CXXFLAGS += -fno-strict-aliasing
endif
ifneq ($(CLANG_MAJOR)$(GCC_MAJOR),)
# Building with clang or GCC
$(PYROOTO) $(PYROOTDO): CXXFLAGS += -Wno-error=format
endif

ifneq ($(CLANG_MAJOR),)
# Building with clang
$(PYROOTO) $(PYROOTDO): CXXFLAGS += -Wno-ignored-attributes
endif
