# Module.mk for x11 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := x11
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

X11DIR       := $(MODDIR)
X11DIRS      := $(X11DIR)/src
X11DIRI      := $(X11DIR)/inc

##### libGX11 #####
X11L         := $(MODDIRI)/LinkDef.h
X11DS        := $(call stripsrc,$(MODDIRS)/G__X11.cxx)
X11DO        := $(X11DS:.cxx=.o)
X11DH        := $(X11DS:.cxx=.h)

X11H1        := $(wildcard $(MODDIRI)/T*.h)
X11H         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
X11S1        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
X11S2        := $(wildcard $(MODDIRS)/*.c)
X11O         := $(call stripsrc,$(X11S1:.cxx=.o) $(X11S2:.c=.o))

X11DEP       := $(X11O:.o=.d) $(X11DO:.o=.d)

X11LIB       := $(LPATH)/libGX11.$(SOEXT)
X11MAP       := $(X11LIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(X11H))
ALLLIBS     += $(X11LIB)
ALLMAPS     += $(X11MAP)

# include all dependency files
INCLUDEFILES += $(X11DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(X11DIRI)/%.h
		cp $< $@

$(X11LIB):      $(X11O) $(X11DO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGX11.$(SOEXT) $@ "$(X11O) $(X11DO)" \
		   "$(X11LIBEXTRA) $(XLIBS)"

$(X11DS):       $(X11H1) $(X11L) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(X11H1) $(X11L)

$(X11MAP):      $(RLIBMAP) $(MAKEFILEDEP) $(X11L)
		$(RLIBMAP) -o $@ -l $(X11LIB) \
		   -d $(X11LIBDEPM) -c $(X11L)

all-$(MODNAME): $(X11LIB) $(X11MAP)

clean-$(MODNAME):
		@rm -f $(X11O) $(X11DO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(X11DEP) $(X11DS) $(X11DH) $(X11LIB) $(X11MAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(PLATFORM),aix5)
$(X11O): CXXFLAGS += -I$(X11DIRI)
endif
$(X11O) $(X11DO): CXXFLAGS += $(X11INCDIR:%=-I%)
