# Module.mk for win32gdk module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 27/11/2001

MODDIR       := win32gdk
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WIN32GDKDIR  := $(MODDIR)
WIN32GDKDIRS := $(WIN32GDKDIR)/src
WIN32GDKDIRI := $(WIN32GDKDIR)/inc

##### libWin32gdk #####
WIN32GDKL    := $(MODDIRI)/LinkDef.h
WIN32GDKDS   := $(MODDIRS)/G__Win32gdk.cxx
WIN32GDKDO   := $(WIN32GDKDS:.cxx=.o)
WIN32GDKDH   := $(WIN32GDKDS:.cxx=.h)

WIN32GDKH1   := $(MODDIRI)/TGWin32.h
WIN32GDKH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WIN32GDKS1   := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WIN32GDKS2   := $(wildcard $(MODDIRS)/*.c)
WIN32GDKO1   := $(WIN32GDKS1:.cxx=.o)
WIN32GDKO2   := $(WIN32GDKS2:.c=.o)
WIN32GDKO    := $(WIN32GDKO1) $(WIN32GDKO2)

WIN32GDKDEP  := $(WIN32GDKO:.o=.d) $(WIN32GDKDO:.o=.d)

WIN32GDKLIB  := $(LPATH)/libWin32gdk.$(SOEXT)

# GDK libraries and DLL's
GDKLIBS      := lib/gdk-1.3.lib lib/glib-1.3.lib
GDKDLLS      := bin/gdk-1.3.dll bin/glib-1.3.dll bin/iconv-1.3.dll

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WIN32GDKH))
ALLLIBS     += $(WIN32GDKLIB)

# include all dependency files
INCLUDEFILES += $(WIN32GDKDEP)

##### local rules #####
include/%.h:    $(WIN32GDKDIRI)/%.h
		cp $< $@

lib/%.lib:      $(WIN32GDKDIR)/gdk/lib/%.lib
		cp $< $@

bin/%.dll:      $(WIN32GDKDIR)/gdk/dll/%.dll
		cp $< $@

$(WIN32GDKLIB): $(WIN32GDKO) $(WIN32GDKDO) $(MAINLIBS) $(WIN32GDKLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libWin32gdk.$(SOEXT) $@ \
		   "$(WIN32GDKO) $(WIN32GDKDO)" "$(WIN32GDKLIBEXTRA)"

$(WIN32GDKDS):  $(WIN32GDKH1) $(WIN32GDKL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WIN32GDKH1) $(WIN32GDKL)

$(WIN32GDKDO):  $(WIN32GDKDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(WIN32GDKDIR)/gdk/inc \
		   -I$(WIN32GDKDIR)/gdk/inc/gdk -I$(WIN32GDKDIR)/gdk/inc/glib \
		   -o $@ -c $<

all-win32gdk:   $(WIN32GDKLIB)

clean-win32gdk:
		@rm -f $(WIN32GDKO) $(WIN32GDKDO)

clean::         clean-win32gdk

distclean-win32gdk: clean-win32gdk
		@rm -f $(WIN32GDKDEP) $(WIN32GDKDS) $(WIN32GDKDH) \
		   $(WIN32GDKLIB) $(GDKLIBS) $(GDKDLLS)

distclean::     distclean-win32gdk

##### extra rules #####
$(WIN32GDKO1): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(WIN32GDKDIR)/gdk/inc \
	   -I$(WIN32GDKDIR)/gdk/inc/gdk -I$(WIN32GDKDIR)/gdk/inc/glib \
	   -o $@ -c $<
