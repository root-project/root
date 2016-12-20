# Module.mk for rootcling_stage1 module
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

# see also ModuleVars.mk

MODNAME      := rootcling_stage1
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTCLING1DIR      := $(MODDIR)
ROOTCLING1DIRS     := $(ROOTCLING1DIR)/src
ROOTCLING1DIRI     := $(ROOTCLING1DIR)/inc
ROOTCLING1DIRR     := $(ROOTCLING1DIR)/res

##### local rules #####

ifneq ($(HOST),)
.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME):

clean-$(MODNAME):

distclean-$(MODNAME):

else # ifneq ($(HOST),)

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(ROOTCLING1EXE): $(ROOTCLING1O) $(FOUNDATIONO) $(DICTGENO) $(CLINGUTILSO) \
	   $(SNPRINTFO) $(STRLCPYO) $(CLINGO)
	$(LD) $(LDFLAGS) $(OSTHREADLIBDIR) $(OSTHREADLIB) -o $@ \
	   $(ROOTCLING1O) $(FOUNDATIONO) $(DICTGENO) $(CLINGUTILSO) \
       $(SNPRINTFO) $(STRLCPYO) $(CLINGO) $(CLINGLIBEXTRA) $(CILIBS)

all-$(MODNAME): $(ROOTCLING1EXE)

clean-$(MODNAME):
	@rm -f $(ROOTCLING1O)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(ROOTCLINGDEP) $(ROOTCLING1EXE) \
	   $(call stripsrc,$(ROOTCLING1DIRS)/*.exp $(ROOTCLING1DIRS)/*.lib)

distclean:: distclean-$(MODNAME)

##### extra rules ######
$(ROOTCLING1O): $(LLVMDEP)
$(ROOTCLING1O): CXXFLAGS += -UR__HAVE_CONFIG -I$(ROOTCLING1DIRR) -I$(DICTGENDIRR)

# the -rdynamic flag is needed on cygwin to make symbols visible to dlsym
ifneq (,$(filter $(ARCH),win32gcc win64gcc))
$(ROOTCLING1EXE): LDFLAGS += -rdynamic
endif

endif # ifneq ($(HOST),)
