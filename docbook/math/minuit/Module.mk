# Module.mk for minuit module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := minuit
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MINUITDIR    := $(MODDIR)
MINUITDIRS   := $(MINUITDIR)/src
MINUITDIRI   := $(MINUITDIR)/inc

##### libMinuit #####
MINUITL      := $(MODDIRI)/LinkDef.h
MINUITDS     := $(call stripsrc,$(MODDIRS)/G__Minuit.cxx)
MINUITDO     := $(MINUITDS:.cxx=.o)
MINUITDH     := $(MINUITDS:.cxx=.h)

MINUITH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MINUITS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MINUITO      := $(call stripsrc,$(MINUITS:.cxx=.o))

MINUITDEP    := $(MINUITO:.o=.d) $(MINUITDO:.o=.d)

MINUITLIB    := $(LPATH)/libMinuit.$(SOEXT)
MINUITMAP    := $(MINUITLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MINUITH))
ALLLIBS     += $(MINUITLIB)
ALLMAPS     += $(MINUITMAP)

# include all dependency files
INCLUDEFILES += $(MINUITDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MINUITDIRI)/%.h
		cp $< $@

$(MINUITLIB):   $(MINUITO) $(MINUITDO) $(ORDER_) $(MAINLIBS) $(MINUITLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMinuit.$(SOEXT) $@ "$(MINUITO) $(MINUITDO)" \
		   "$(MINUITLIBEXTRA)"

$(MINUITDS):    $(MINUITH) $(MINUITL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MINUITH) $(MINUITL)

$(MINUITMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(MINUITL)
		$(RLIBMAP) -o $@ -l $(MINUITLIB) \
		   -d $(MINUITLIBDEPM) -c $(MINUITL)

all-$(MODNAME): $(MINUITLIB) $(MINUITMAP)

clean-$(MODNAME):
		@rm -f $(MINUITO) $(MINUITDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MINUITDEP) $(MINUITDS) $(MINUITDH) $(MINUITLIB) $(MINUITMAP)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(MINUITDO): NOOPT = $(OPT)
