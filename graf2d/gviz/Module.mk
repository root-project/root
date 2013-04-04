# Module.mk for gviz module
# Copyright (c) 2009 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 2/9/2009

MODNAME      := gviz
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GVIZDIR      := $(MODDIR)
GVIZDIRS     := $(GVIZDIR)/src
GVIZDIRI     := $(GVIZDIR)/inc

##### libGviz #####
GVIZL        := $(MODDIRI)/LinkDef.h
GVIZDS       := $(call stripsrc,$(MODDIRS)/G__Gviz.cxx)
GVIZDO       := $(GVIZDS:.cxx=.o)
GVIZDH       := $(GVIZDS:.cxx=.h)

GVIZH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GVIZS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GVIZO        := $(call stripsrc,$(GVIZS:.cxx=.o))

GVIZDEP      := $(GVIZO:.o=.d) $(GVIZDO:.o=.d)

GVIZLIB      := $(LPATH)/libGviz.$(SOEXT)
GVIZMAP      := $(GVIZLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GVIZH))
ALLLIBS     += $(GVIZLIB)
ALLMAPS     += $(GVIZMAP)

# include all dependency files
INCLUDEFILES += $(GVIZDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GVIZDIRI)/%.h
		cp $< $@

$(GVIZLIB):     $(GVIZO) $(GVIZDO) $(ORDER_) $(MAINLIBS) $(GVIZLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGviz.$(SOEXT) $@ "$(GVIZO) $(GVIZDO)" \
		   "$(GVIZLIBEXTRA) $(GRAPHVIZLIB)"

$(GVIZDS):      $(GVIZH) $(GVIZL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GVIZH) $(GVIZL)

$(GVIZMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(GVIZL)
		$(RLIBMAP) -o $@ -l $(GVIZLIB) \
		   -d $(GVIZLIBDEPM) -c $(GVIZL)

all-$(MODNAME): $(GVIZLIB) $(GVIZMAP)

clean-$(MODNAME):
		@rm -f $(GVIZO) $(GVIZDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GVIZDEP) $(GVIZDS) $(GVIZDH) $(GVIZLIB) $(GVIZMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GVIZO) $(GVIZDO): CXXFLAGS += $(GRAPHVIZINCDIR:%=-I%) $(GRAPHVIZCFLAGS)
