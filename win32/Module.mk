# Module.mk for win32 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := win32
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WIN32DIR     := $(MODDIR)
WIN32DIRS    := $(WIN32DIR)/src
WIN32DIRI    := $(WIN32DIR)/inc

##### libWin32 #####
WIN32L       := $(MODDIRI)/LinkDef.h
WIN32DS      := $(MODDIRS)/G__Win32.cxx
WIN32DO      := $(WIN32DS:.cxx=.o)
WIN32DH      := $(WIN32DS:.cxx=.h)

WIN32H1      := $(MODDIRI)/TGWin32.h $(MODDIRI)/TWin32GuiFactory.h
WIN32H       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WIN32S       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WIN32O       := $(WIN32S:.cxx=.o)

WIN32DEP     := $(WIN32O:.o=.d) $(WIN32DO:.o=.d)

WIN32LIB     := $(LPATH)/libWin32.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WIN32H))
ALLLIBS     += $(WIN32LIB)

# include all dependency files
INCLUDEFILES += $(WIN32DEP)

##### local rules #####
include/%.h:    $(WIN32DIRI)/%.h
		cp $< $@

$(WIN32LIB):    $(WIN32O) $(WIN32DO) $(MAINLIBS) $(WIN32LIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libWin32.$(SOEXT) $@ "$(WIN32O) $(WIN32DO)" \
		   "$(WIN32LIBEXTRA)"

$(WIN32DS):     $(WIN32H1) $(WIN32L) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WIN32H1) $(WIN32L)

$(WIN32DO):     $(WIN32DS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-win32:      $(WIN32LIB)

map-win32:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(WIN32LIB) \
		   -d $(WIN32LIBDEP) -c $(WIN32L)

map::           map-win32

clean-win32:
		@rm -f $(WIN32O) $(WIN32DO)

clean::         clean-win32

distclean-win32: clean-win32
		@rm -f $(WIN32DEP) $(WIN32DS) $(WIN32DH) $(WIN32LIB)

distclean::     distclean-win32
