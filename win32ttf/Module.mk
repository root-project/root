# Module.mk for win32ttf module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := win32ttf
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WIN32TTFDIR  := $(MODDIR)
WIN32TTFDIRS := $(WIN32TTFDIR)/src
WIN32TTFDIRI := $(WIN32TTFDIR)/inc

WIN32GDKDIR  := win32gdk

##### libGWin32TTF #####
WIN32TTFL    := $(MODDIRI)/LinkDef.h
WIN32TTFDS   := $(MODDIRS)/G__Win32TTF.cxx
WIN32TTFDO   := $(WIN32TTFDS:.cxx=.o)
WIN32TTFDH   := $(WIN32TTFDS:.cxx=.h)

WIN32TTFH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WIN32TTFS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WIN32TTFO    := $(WIN32TTFS:.cxx=.o)

WIN32TTFDEP  := $(WIN32TTFO:.o=.d) $(WIN32TTFDO:.o=.d)

WIN32TTFLIB  := $(LPATH)/libGWin32TTF.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WIN32TTFH))
ALLLIBS     += $(WIN32TTFLIB)

# include all dependency files
INCLUDEFILES += $(WIN32TTFDEP)

##### local rules #####
include/%.h:    $(WIN32TTFDIRI)/%.h
		cp $< $@

$(WIN32TTFLIB): $(WIN32TTFO) $(WIN32TTFDO) $(FREETYPELIB) $(MAINLIBS) $(WIN32TTFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGWin32TTF.$(SOEXT) $@ \
		   "$(WIN32TTFO) $(WIN32TTFDO)" \
		   "$(FREETYPELIB) $(WIN32TTFLIBEXTRA)"

$(WIN32TTFDS):  $(WIN32TTFH) $(WIN32TTFL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WIN32TTFH) $(WIN32TTFL)

$(WIN32TTFDO):  $(WIN32TTFDS) $(FREETYPELIB)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(FREETYPEDIRI) \
		   -I$(WIN32GDKDIR)/gdk/inc -I$(WIN32GDKDIR)/gdk/inc/gdk \
		   -I$(WIN32GDKDIR)/gdk/inc/glib -o $@ -c $<

all-win32ttf:   $(WIN32TTFLIB)

clean-win32ttf:
		@rm -f $(WIN32TTFO) $(WIN32TTFDO)

clean::         clean-win32ttf

distclean-win32ttf: clean-win32ttf
		@rm -f $(WIN32TTFDEP) $(WIN32TTFDS) $(WIN32TTFDH) $(WIN32TTFLIB)

distclean::     distclean-win32ttf

##### extra rules ######
$(WIN32TTFO): %.o: %.cxx $(FREETYPELIB)
	$(CXX) $(OPT) $(CXXFLAGS) -I$(FREETYPEDIRI) \
	   -I$(WIN32GDKDIR)/gdk/inc -I$(WIN32GDKDIR)/gdk/inc/gdk \
	   -I$(WIN32GDKDIR)/gdk/inc/glib -o $@ -c $<
