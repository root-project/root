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
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HDFSH))
ALLLIBS     += $(HDFSLIB)
ALLMAPS     += $(HDFSMAP)

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

$(HDFSDS):      $(HDFSH) $(HDFSL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HDFSH) $(HDFSL)

$(HDFSMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(HDFSL)
		$(RLIBMAP) -o $@ -l $(HDFSLIB) \
		   -d $(HDFSLIBDEPM) -c $(HDFSL)

all-$(MODNAME): $(HDFSLIB) $(HDFSMAP)

clean-$(MODNAME):
		@rm -f $(HDFSO) $(HDFSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HDFSDEP) $(HDFSDS) $(HDFSDH) $(HDFSLIB) $(HDFSMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(HDFSO) $(HDFSDO): CXXFLAGS += $(HDFSCFLAGS) $(HDFSINCDIR:%=-I%) $(JNIINCDIR:%=-I%)
