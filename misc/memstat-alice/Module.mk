# Module.mk for newdelete module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Anar Manafov 17/06/2008

MODNAME       := memstat
MODDIR        := misc/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

MEMSTATDIR    := $(MODDIR)
MEMSTATDIRS   := $(MEMSTATDIR)/src
MEMSTATDIRI   := $(MEMSTATDIR)/inc

##### libMemStat #####
MEMSTATL      := $(MODDIRI)/LinkDef.h
MEMSTATDS     := $(MODDIRS)/G__Memstat.cxx
MEMSTATDO     := $(MEMSTATDS:.cxx=.o)
MEMSTATDH     := $(MEMSTATDS:.cxx=.h)

MEMSTATH      := $(MODDIRI)/TMemStatHelpers.h $(MODDIRI)/TMemStatDepend.h \
                 $(MODDIRI)/TMemStat.h \
		 $(MODDIRI)/TMemStatManager.h $(MODDIRI)/TMemStatInfo.h
MEMSTATS      := $(MODDIRS)/TMemStat.cxx $(MODDIRS)/TMemStatManager.cxx \
		 $(MODDIRS)/TMemStatDepend.cxx $(MODDIRS)/TMemStatInfo.cxx \
		 $(MODDIRS)/TMemStatHelpers.cxx
MEMSTATO      := $(MEMSTATS:.cxx=.o)

MEMSTATDEP    := $(MEMSTATO:.o=.d) $(MEMSTATDO:.o=.d)

MEMSTATLIB    := $(LPATH)/libMemStat.$(SOEXT)
MEMSTATMAP    := $(MEMSTATLIB:.$(SOEXT)=.rootmap)

##### libMemStatGui #####
MEMSTATGUIL   := $(MODDIRI)/LinkDefGUI.h
MEMSTATGUIDS  := $(MODDIRS)/G__MemstatGui.cxx
MEMSTATGUIDO  := $(MEMSTATGUIDS:.cxx=.o)
MEMSTATGUIDH  := $(MEMSTATGUIDS:.cxx=.h)

MEMSTATGUIH   := $(MODDIRI)/TMemStat.h $(MODDIRI)/TMemStatViewerGUI.h \
                 $(MODDIRI)/TMemStatDrawDlg.h $(MODDIRI)/TMemStatResource.h
MEMSTATGUIS   := $(MODDIRS)/TMemStatViewerGUI.cxx $(MODDIRS)/TMemStatDrawDlg.cxx
MEMSTATGUIO   := $(MEMSTATGUIS:.cxx=.o)

MEMSTATGUIDEP := $(MEMSTATGUIO:.o=.d) $(MEMSTATGUIDO:.o=.d)

MEMSTATGUILIB := $(LPATH)/libMemStatGui.$(SOEXT)
MEMSTATGUIMAP := $(MEMSTATGUILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MEMSTATH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MEMSTATGUIH))
ALLLIBS     += $(MEMSTATLIB) $(MEMSTATGUILIB)
ALLMAPS     += $(MEMSTATMAP) $(MEMSTATGUIMAP)

# include all dependency files
INCLUDEFILES += $(MEMSTATDEP) $(MEMSTATGUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MEMSTATDIRI)/%.h
		cp $< $@

##### libMemStat #####
$(MEMSTATLIB):  $(MEMSTATO) $(MEMSTATDO) $(ORDER_) $(MAINLIBS) $(MEMSTATLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMemStat.$(SOEXT) $@ \
		   "$(MEMSTATO) $(MEMSTATDO)" "$(MEMSTATLIBEXTRA)"

$(MEMSTATDS):   $(MEMSTATH) $(MEMSTATL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MEMSTATH) $(MEMSTATL)

$(MEMSTATMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(MEMSTATL)
		$(RLIBMAP) -o $@ -l $(MEMSTATLIB) \
		   -d $(MEMSTATLIBDEPM) -c $(MEMSTATL)

##### libMemStatGui #####
$(MEMSTATGUILIB): $(MEMSTATGUIO) $(MEMSTATGUIDO) $(ORDER_) \
                  $(MAINLIBS) $(MEMSTATGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMemStatGui.$(SOEXT) $@ \
		   "$(MEMSTATGUIO) $(MEMSTATGUIDO)" "$(MEMSTATGUILIBEXTRA)"

$(MEMSTATGUIDS): $(MEMSTATGUIH) $(MEMSTATGUIL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MEMSTATGUIH) $(MEMSTATGUIL)

$(MEMSTATGUIMAP): $(RLIBMAP) $(MAKEFILEDEP) $(MEMSTATGUIL)
		$(RLIBMAP) -o $(MEMSTATGUIMAP) -l $(MEMSTATGUILIB) \
		   -d $(MEMSTATGUILIBDEPM) -c $(MEMSTATGUIL)

all-$(MODNAME): $(MEMSTATLIB) $(MEMSTATMAP) $(MEMSTATGUILIB) $(MEMSTATGUIMAP)

clean-$(MODNAME):
		@rm -f $(MEMSTATO) $(MEMSTATDO) $(MEMSTATGUIO) $(MEMSTATGUIDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MEMSTATDEP) $(MEMSTATDS) $(MEMSTATDH) $(MEMSTATLIB) \
		   $(MEMSTATMAP) $(MEMSTATGUIDEP) $(MEMSTATGUIDS) \
		   $(MEMSTATGUIDH) $(MEMSTATGUILIB) $(MEMSTATGUIMAP)

distclean::     distclean-$(MODNAME)
