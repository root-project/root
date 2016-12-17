# Module.mk for hdfs module
# Copyright (c) 2009 Rene Brun and Fons Rademakers
#
# Author: Brian Bockelman, 29/9/2009

MODNAME      := hdfs
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HDFSDIR      := $(MODDIR)
HDFSDIRS     := $(HDFSDIR)/src
HDFSDIRI     := $(HDFSDIR)/inc

##### libHDFS #####
HDFSL        := $(MODDIRI)/LinkDef.h
HDFSDS       := $(call stripsrc,$(MODDIRS)/G__HDFS.cxx)
HDFSDO       := $(HDFSDS:.cxx=.o)
HDFSDH       := $(HDFSDS:.cxx=.h)

HDFSH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HDFSS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HDFSO        := $(call stripsrc,$(HDFSS:.cxx=.o))

HDFSDEP      := $(HDFSO:.o=.d) $(HDFSDO:.o=.d)

HDFSLIB      := $(LPATH)/libHDFS.$(SOEXT)
HDFSMAP      := $(HDFSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
HDFSH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(HDFSH))
ALLHDRS     += $(HDFSH_REL)
ALLLIBS     += $(HDFSLIB)
ALLMAPS     += $(HDFSMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(HDFSH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Io_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(HDFSLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(HDFSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HDFSDIRI)/%.h
		cp $< $@

$(HDFSLIB):     $(HDFSO) $(HDFSDO) $(ORDER_) $(MAINLIBS) $(HDFSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHDFS.$(SOEXT) $@ "$(HDFSO) $(HDFSDO)" \
		   "$(HDFSLIBEXTRA) $(HDFSLIBDIR) $(HDFSCLILIB) $(JVMLIBDIR) $(JVMCLILIB)"

$(call pcmrule,HDFS)
	$(noop)

$(HDFSDS):      $(HDFSH) $(HDFSL) $(ROOTCLINGEXE) $(call pcmdep,HDFS)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,HDFS) -c $(HDFSH) $(HDFSL)

$(HDFSMAP):     $(HDFSH) $(HDFSL) $(ROOTCLINGEXE) $(call pcmdep,HDFS)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HDFSDS) $(call dictModule,HDFS) -c $(HDFSH) $(HDFSL)

all-$(MODNAME): $(HDFSLIB)

clean-$(MODNAME):
		@rm -f $(HDFSO) $(HDFSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HDFSDEP) $(HDFSDS) $(HDFSDH) $(HDFSLIB) $(HDFSMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(HDFSO) $(HDFSDO): CXXFLAGS += $(HDFSCFLAGS) $(HDFSINCDIR:%=-I%) $(JNIINCDIR:%=-I%)
