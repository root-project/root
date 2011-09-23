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

##### python64 #####
ifeq ($(ARCH),macosx64)
ifeq ($(MACOSX_MINOR),5)
PYTHON64S    := $(MODDIRS)/python64.c
PYTHON64O    := $(call stripsrc,$(PYTHON64S:.c=.o))
PYTHON64     := bin/python64
PYTHON64DEP  := $(PYTHON64O:.o=.d)
endif
endif

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
ifeq ($(ARCH),win32)
ROOTPY       := $(subst $(MODDIR),bin,$(ROOTPYS))
bin/%.py: $(MODDIR)/%.py; cp $< $@
else
ROOTPY       := $(subst $(MODDIR),$(LPATH),$(ROOTPYS))
$(LPATH)/%.py: $(MODDIR)/%.py; cp $< $@
endif
ROOTPYC      := $(ROOTPY:.py=.pyc)
ROOTPYO      := $(ROOTPY:.py=.pyo)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYROOTH))
ALLLIBS      += $(PYROOTLIB)
ALLMAPS      += $(PYROOTMAP)

ALLEXECS     += $(PYTHON64)
INCLUDEFILES += $(PYTHON64DEP)

# include all dependency files
INCLUDEFILES += $(PYROOTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PYROOTDIRI)/%.h
		cp $< $@

%.pyc: %.py;    python -c 'import py_compile; py_compile.compile( "$<" )'
%.pyo: %.py;    python -O -c 'import py_compile; py_compile.compile( "$<" )'

$(PYROOTLIB):   $(PYROOTO) $(PYROOTDO) $(ROOTPY) $(ROOTPYC) $(ROOTPYO) \
                $(ROOTLIBSDEP) $(PYTHONLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		  "$(SOFLAGS)" libPyROOT.$(SOEXT) $@ \
		  "$(PYROOTO) $(PYROOTDO)" \
		  "$(ROOTULIBS) $(RPATH) $(ROOTLIBS) $(PYROOTLIBEXTRA) \
		   $(PYTHONLIBDIR) $(PYTHONLIB)" "$(PYTHONLIBFLAGS)"
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

$(PYROOTDS):    $(PYROOTH) $(PYROOTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PYROOTH) $(PYROOTL)

$(PYROOTMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(PYROOTL)
		$(RLIBMAP) -o $@ -l $(PYROOTLIB) \
		   -d $(PYROOTLIBDEPM) -c $(PYROOTL)

$(PYTHON64):    $(PYTHON64O)
		$(CC) $(LDFLAGS) -o $@ $(PYTHON64O) \
		   $(PYTHONLIBDIR) $(PYTHONLIB)

all-$(MODNAME): $(PYROOTLIB) $(PYROOTMAP) $(PYTHON64)

clean-$(MODNAME):
		@rm -f $(PYROOTO) $(PYROOTDO) $(PYTHON64O)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PYROOTDEP) $(PYROOTDS) $(PYROOTDH) $(PYROOTLIB) \
		   $(ROOTPY) $(ROOTPYC) $(ROOTPYO) $(PYROOTMAP) \
		   $(PYROOTPYD) $(PYTHON64DEP) $(PYTHON64)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PYROOTO): CXXFLAGS += $(PYTHONINCDIR:%=-I%)
$(PYTHON64O): CFLAGS += $(PYTHONINCDIR:%=-I%)
ifeq ($(GCC_MAJOR),4)
$(PYROOTO): CXXFLAGS += -fno-strict-aliasing
endif
