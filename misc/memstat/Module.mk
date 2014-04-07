# Module.mk for newdelete module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Anar Manafov 17/06/2008

MODNAME       := memstat
MODDIR        := $(ROOT_SRCDIR)/misc/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

MEMSTATDIR    := $(MODDIR)
MEMSTATDIRS   := $(MEMSTATDIR)/src
MEMSTATDIRI   := $(MEMSTATDIR)/inc

##### libMemStat #####
MEMSTATL      := $(MODDIRI)/LinkDef.h
MEMSTATDS     := $(call stripsrc,$(MODDIRS)/G__MemStat.cxx)
MEMSTATDO     := $(MEMSTATDS:.cxx=.o)
MEMSTATDH     := $(MEMSTATDS:.cxx=.h)

MEMSTATH      := $(MODDIRI)/TMemStatHelpers.h \
                 $(MODDIRI)/TMemStat.h $(MODDIRI)/TMemStatBacktrace.h \
                 $(MODDIRI)/TMemStatDef.h \
		 $(MODDIRI)/TMemStatMng.h $(MODDIRI)/TMemStatHook.h

MEMSTATS      := $(MODDIRS)/TMemStat.cxx $(MODDIRS)/TMemStatMng.cxx \
		 $(MODDIRS)/TMemStatBacktrace.cxx \
		 $(MODDIRS)/TMemStatHelpers.cxx $(MODDIRS)/TMemStatHook.cxx
MEMSTATO      := $(call stripsrc,$(MEMSTATS:.cxx=.o))

MEMSTATDEP    := $(MEMSTATO:.o=.d) $(MEMSTATDO:.o=.d)

MEMSTATLIB    := $(LPATH)/libMemStat.$(SOEXT)
MEMSTATMAP    := $(MEMSTATLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MEMSTATH))
ALLLIBS     += $(MEMSTATLIB)
ALLMAPS     += $(MEMSTATMAP)
  
# include all dependency files
INCLUDEFILES += $(MEMSTATDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MEMSTATDIRI)/%.h
		cp $< $@

##### libMemStat #####
$(MEMSTATLIB):  $(MEMSTATO) $(MEMSTATDO) $(ORDER_) $(MAINLIBS) $(MEMSTATLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMemStat.$(SOEXT) $@ \
		   "$(MEMSTATO) $(MEMSTATDO)" "$(MEMSTATLIBEXTRA)"

$(call pcmrule,MEMSTAT)
	$(noop)

$(MEMSTATDS):   $(MEMSTATH) $(MEMSTATL) $(ROOTCLINGEXE) $(call pcmdep,MEMSTAT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,MEMSTAT) -c $(MEMSTATH) $(MEMSTATL)

$(MEMSTATMAP):  $(MEMSTATH) $(MEMSTATL) $(ROOTCLINGEXE) $(call pcmdep,MEMSTAT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(MEMSTATDS) $(call dictModule,MEMSTAT) -c $(MEMSTATH) $(MEMSTATL)


all-$(MODNAME): $(MEMSTATLIB)

clean-$(MODNAME):
		@rm -f $(MEMSTATO) $(MEMSTATDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MEMSTATDEP) $(MEMSTATDS) $(MEMSTATDH) $(MEMSTATLIB) \
		   $(MEMSTATMAP) $(LPATH)/libMemStatGui.$(SOEXT) \
		   $(LPATH)/libMemStatGui.rootmap

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(GLIBC_MALLOC_DEPRECATED),yes)
$(MEMSTATO) $(MEMSTATDO): CXXFLAGS += -Wno-deprecated-declarations
endif
