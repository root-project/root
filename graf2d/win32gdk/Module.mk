# Module.mk for win32gdk module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 27/11/2001

MODNAME      := win32gdk
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WIN32GDKDIR  := $(MODDIR)
WIN32GDKDIRS := $(WIN32GDKDIR)/src
WIN32GDKDIRI := $(WIN32GDKDIR)/inc

GDKVERS      := gdk/src
GDKDIRS      := $(MODDIR)/$(GDKVERS)/gdk
GDKDIRI      := $(MODDIR)/$(GDKVERS)/gdk
GLIBDIRI     := $(MODDIR)/$(GDKVERS)/glib

##### gdk-1.3.dll #####
GDKDLLA      := $(call stripsrc,$(GDKDIRS)/gdk-1.3.dll)
GDKLIBA      := $(call stripsrc,$(GDKDIRS)/gdk-1.3.lib)
GDKDLL       := bin/gdk-1.3.dll
GDKLIB       := $(LPATH)/gdk-1.3.lib
GDKSRC       := $(wildcard $(GDKDIRS)/*.c) $(wildcard $(GDKDIRS)/win32/*.c)

GDKNMCXXFLAGS:= "$(OPT) $(BLDCXXFLAGS) -I$(shell cygpath -w '$(GLIBDIRI)') -FI$(shell cygpath -w '$(ROOT_SRCDIR)/build/win/w32pragma.h')"
ifeq (yes,$(WINRTDEBUG))
GDKNMCXXFLAGS += DEBUG=1
endif

##### libWin32gdk #####
WIN32GDKL    := $(MODDIRI)/LinkDef.h
WIN32GDKDS   := $(call stripsrc,$(MODDIRS)/G__Win32gdk.cxx)
WIN32GDKDO   := $(WIN32GDKDS:.cxx=.o)
WIN32GDKDH   := $(WIN32GDKDS:.cxx=.h)

WIN32GDKH1   := $(MODDIRI)/TGWin32.h $(MODDIRI)/TGWin32GL.h
WIN32GDKH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WIN32GDKS1   := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WIN32GDKS2   := $(wildcard $(MODDIRS)/*.c)
WIN32GDKO1   := $(call stripsrc,$(WIN32GDKS1:.cxx=.o))
WIN32GDKO2   := $(call stripsrc,$(WIN32GDKS2:.c=.o))
WIN32GDKO    := $(WIN32GDKO1) $(WIN32GDKO2)

WIN32GDKDEP  := $(WIN32GDKO:.o=.d) $(WIN32GDKDO:.o=.d)

WIN32GDKLIB  := $(LPATH)/libWin32gdk.$(SOEXT)
WIN32GDKMAP  := $(WIN32GDKLIB:.$(SOEXT)=.rootmap)

# GDK libraries and DLL's
GDKLIBS      := lib/glib-1.3.lib
GDKDLLS      := bin/glib-1.3.dll bin/iconv-1.3.dll

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WIN32GDKH))
ALLLIBS     += $(WIN32GDKLIB)
ALLMAPS     += $(WIN32GDKMAP)

# include all dependency files
INCLUDEFILES += $(WIN32GDKDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(WIN32GDKDIRI)/%.h
		cp $< $@

lib/%.lib:      $(WIN32GDKDIR)/gdk/lib/%.lib
		cp $< $@

bin/%.dll:      $(WIN32GDKDIR)/gdk/dll/%.dll
		cp $< $@

$(GDKLIB):      $(GDKLIBA)
		cp $< $@

$(GDKDLL):      $(GDKLIBA)
		cp $(GDKDLLA) $@

$(GDKLIBA):     $(GDKSRC)
		$(MAKEDIR)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.lib' --exclude '*.dll' $(GDKDIRS) $(dir $(call stripsrc,$(GDKDIRS)))
		@$(RSYNC) --exclude '.svn' $(WIN32GDKDIR)/gdk/lib $(call stripsrc,$(WIN32GDKDIR)/gdk)
endif
		@(echo "*** Building $@..."; \
		  unset MAKEFLAGS; \
		  cd $(call stripsrc,$(GDKDIRS)/win32); \
		  nmake -nologo -f makefile.msc \
		  NMCXXFLAGS=$(GDKNMCXXFLAGS) VC_MAJOR=$(VC_MAJOR); \
		  cd ..; \
		  nmake -nologo -f makefile.msc \
                  NMCXXFLAGS=$(GDKNMCXXFLAGS) VC_MAJOR=$(VC_MAJOR))

$(WIN32GDKLIB): $(WIN32GDKO) $(WIN32GDKDO) $(FREETYPEDEP) $(GDKLIB) $(GDKDLL) \
                $(ORDER_) $(MAINLIBS) $(GDKDLLS) $(GDKLIBS) $(WIN32GDKLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libWin32gdk.$(SOEXT) $@ \
		   "$(WIN32GDKO) $(WIN32GDKDO)" \
		   "$(FREETYPELDFLAGS) $(FREETYPELIB) $(GDKLIB) $(WIN32GDKLIBEXTRA) $(GDKLIBS) Glu32.lib Opengl32.lib"

$(WIN32GDKDS):  $(WIN32GDKH1) $(WIN32GDKL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(WIN32GDKH1) $(WIN32GDKL)

$(WIN32GDKMAP): $(RLIBMAP) $(MAKEFILEDEP) $(WIN32GDKL)
		$(RLIBMAP) -o $@ -l $(WIN32GDKLIB) \
		   -d $(WIN32GDKLIBDEPM) -c $(WIN32GDKL)

all-$(MODNAME): $(WIN32GDKLIB) $(WIN32GDKMAP)

clean-$(MODNAME):
		@rm -f $(WIN32GDKO) $(WIN32GDKDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(WIN32GDKDEP) $(WIN32GDKDS) $(WIN32GDKDH) \
		   $(WIN32GDKLIB) $(WIN32GDKMAP) $(GDKLIBS) $(GDKDLLS)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(call stripsrc,$(GDKDIRS))
		@rm -rf $(call stripsrc,$(WIN32GDKDIR)/gdk/lib)
else
		-@(cd $(call stripsrc,$(GDKDIRS)); unset MAKEFLAGS; \
		nmake -nologo -f makefile.msc clean VC_MAJOR=$(VC_MAJOR))
endif

distclean::     distclean-$(MODNAME)

##### extra rules #####
$(WIN32GDKO1) $(WIN32GDKDO): $(FREETYPEDEP)
$(WIN32GDKO1) $(WIN32GDKDO): CXXFLAGS += $(FREETYPEINC) \
  -I$(WIN32GDKDIR)/gdk/src $(GDKDIRI:%=-I%) $(GLIBDIRI:%=-I%)
