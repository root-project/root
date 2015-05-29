# Module.mk for rtools module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Kirby Hermansen and Lorenzo Moneta, 23/07/2014 

MODNAME      := rtools
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RTOOLSDIR  := $(MODDIR)
RTOOLSDIRS := $(RTOOLSDIR)/src
RTOOLSDIRI := $(RTOOLSDIR)/inc/Math
RTOOLSDIRT := $(call stripsrc,$(RTOOLSDIR)/test)

##### libRtools #####
RTOOLSL    := $(MODDIRI)/LinkDef.h
RTOOLSDS   := $(call stripsrc,$(MODDIRS)/G__RMinimizer.cxx)
RTOOLSDO   := $(RTOOLSDS:.cxx=.o)
RTOOLSDH   := $(RTOOLSDS:.cxx=.h)

RTOOLSDH1  :=  $(MODDIRI)/Math/RMinimizer.h

RTOOLSH    := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
RTOOLSS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RTOOLSO    := $(call stripsrc,$(RTOOLSS:.cxx=.o))

RTOOLSDEP  := $(RTOOLSO:.o=.d) $(RTOOLSDO:.o=.d)

RTOOLSLIB  := $(LPATH)/libRtools.$(SOEXT)
RTOOLSMAP  := $(RTOOLSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(RTOOLSH))
ALLLIBS      += $(RTOOLSLIB)
ALLMAPS      += $(RTOOLSMAP)

# include all dependency files
INCLUDEFILES += $(RTOOLSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/Math/%.h: $(RTOOLSDIRI)/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

$(RTOOLSLIB): $(RTOOLSO) $(RTOOLSDO) $(ORDER_) $(MAINLIBS) $(RTOOLSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libRtools.$(SOEXT) $@     \
		   "$(RTOOLSO) $(RTOOLSDO)" \
		   "$(RTOOLSLIBEXTRA) $(RLIBS)"

$(RTOOLSDS):  $(RTOOLSDH1) $(RTOOLSL) $(RTOOLSLINC) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(RTOOLSDH1) $(RTOOLSL)

$(RTOOLSMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(RTOOLSL) $(RTOOLSLINC)
		$(RLIBMAP) -o $@ -l $(RTOOLSLIB) \
		   -d $(RTOOLSLIBDEPM) -c $(RTOOLSL) $(RTOOLSLINC)

all-$(MODNAME): $(RTOOLSLIB) $(RTOOLSMAP)

clean-$(MODNAME):
		@rm -f $(RTOOLSO) $(RTOOLSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RTOOLSDEP) $(RTOOLSDS) $(RTOOLSDH) \
		   $(RTOOLSLIB) $(RTOOLSMAP)
		@rm -rf include/Math
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(RTOOLSDIRT)
else
		@cd $(RTOOLSDIRT) && $(MAKE) distclean
endif

distclean::     distclean-$(MODNAME)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(RTOOLSDIR)/test $(RTOOLSDIRT)
endif
		@cd $(RTOOLSDIRT) && $(MAKE)

##### extra rules ######
$(RTOOLSO):   CXXFLAGS += $(RFLAGS)

$(RTOOLSDO):   CXXFLAGS += $(RFLAGS)

