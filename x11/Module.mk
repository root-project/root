# Module.mk for x11 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := x11
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

X11DIR       := $(MODDIR)
X11DIRS      := $(X11DIR)/src
X11DIRI      := $(X11DIR)/inc

##### libGX11 #####
X11L         := $(MODDIRI)/LinkDef.h
X11DS        := $(MODDIRS)/G__X11.cxx
X11DO        := $(X11DS:.cxx=.o)
X11DH        := $(X11DS:.cxx=.h)

X11H1        := $(wildcard $(MODDIRI)/T*.h)
X11H         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
X11S1        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
X11S2        := $(wildcard $(MODDIRS)/*.c)
X11O         := $(X11S1:.cxx=.o) $(X11S2:.c=.o)

X11DEP       := $(X11O:.o=.d) $(X11DO:.o=.d)

X11LIB       := $(LPATH)/libGX11.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(X11H))
ALLLIBS     += $(X11LIB)

# include all dependency files
INCLUDEFILES += $(X11DEP)

##### local rules #####
include/%.h:    $(X11DIRI)/%.h
		cp $< $@

$(X11LIB):      $(X11O) $(X11DO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGX11.$(SOEXT) $@ "$(X11O) $(X11DO)" \
		   "$(X11LIBEXTRA) $(XLIBS)"

$(X11DS):       $(X11H1) $(X11L) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(X11H1) $(X11L)

$(X11DO):       $(X11DS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-x11:        $(X11LIB)

map-x11:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(X11LIB) \
		   -d $(X11LIBDEP) -c $(X11L)

map::           map-x11

clean-x11:
		@rm -f $(X11O) $(X11DO)

clean::         clean-x11

distclean-x11:  clean-x11
		@rm -f $(X11DEP) $(X11DS) $(X11DH) $(X11LIB)

distclean::     distclean-x11
