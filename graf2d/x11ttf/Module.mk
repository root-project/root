# Module.mk for x11ttf module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := x11ttf
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

X11TTFDIR    := $(MODDIR)
X11TTFDIRS   := $(X11TTFDIR)/src
X11TTFDIRI   := $(X11TTFDIR)/inc

##### libGX11TTF #####
X11TTFL      := $(MODDIRI)/LinkDef.h
X11TTFDS     := $(call stripsrc,$(MODDIRS)/G__GX11TTF.cxx)
X11TTFDO     := $(X11TTFDS:.cxx=.o)
X11TTFDH     := $(X11TTFDS:.cxx=.h)

X11TTFH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
X11TTFS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
X11TTFO      := $(call stripsrc,$(X11TTFS:.cxx=.o))

X11TTFDEP    := $(X11TTFO:.o=.d) $(X11TTFDO:.o=.d)

X11TTFLIB    := $(LPATH)/libGX11TTF.$(SOEXT)
X11TTFMAP    := $(X11TTFLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(X11TTFH))
ALLLIBS     += $(X11TTFLIB)
ALLMAPS     += $(X11TTFMAP)

# include all dependency files
INCLUDEFILES += $(X11TTFDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(X11TTFDIRI)/%.h
		cp $< $@

$(X11TTFLIB):   $(X11TTFO) $(X11TTFDO) $(FREETYPEDEP) $(ORDER_) $(MAINLIBS) \
                $(X11TTFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGX11TTF.$(SOEXT) $@ \
		   "$(X11TTFO) $(X11TTFDO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(X11TTFLIBEXTRA) $(XLIBS)"

$(call pcmrule,X11TTF)
	$(noop)

$(X11TTFDS):    $(X11TTFH) $(X11TTFL) $(ROOTCLINGEXE) $(call pcmdep,X11TTF)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,X11TTF) -c -writeEmptyRootPCM $(FREETYPEINC) $(X11TTFH) $(X11TTFL)

$(X11TTFMAP):   $(X11TTFH) $(X11TTFL) $(ROOTCLINGEXE) $(call pcmdep,X11TTF)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(X11TTFDS) $(call dictModule,X11TTF) -c $(FREETYPEINC) $(X11TTFH) $(X11TTFL)

all-$(MODNAME): $(X11TTFLIB)

clean-$(MODNAME):
		@rm -f $(X11TTFO) $(X11TTFDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(X11TTFDEP) $(X11TTFDS) $(X11TTFDH) $(X11TTFLIB) \
		   $(X11TTFMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(X11TTFO) $(X11TTFDO): $(FREETYPEDEP)
ifeq ($(PLATFORM),macosx)
$(X11TTFO) $(X11TTFDO): CXXFLAGS += $(X11INCDIR:%=-I%) $(FREETYPEINC)
else
ifeq ($(PLATFORM),aix5)
$(X11TTFO) $(X11TTFDO): CXXFLAGS += -I$(X11DIRI) $(FREETYPEINC)
else
$(X11TTFO) $(X11TTFDO): CXXFLAGS += $(FREETYPEINC)
endif
endif
ifeq ($(XFTLIB),yes)
$(X11TTFLIB): XLIBS += $(X11LIBDIR) -lXft
endif
