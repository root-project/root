# Module.mk for webgui module
# Copyright (c) 2016 Sergey Linev
#
# Author: Sergey Linev, 7/12/2016

MODNAME      := web
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

WEBGUIDIR    := $(MODDIR)
WEBGUIDIRS   := $(WEBGUIDIR)/src
WEBGUIDIRI   := $(WEBGUIDIR)/inc

##### libWebGui #####
WEBGUIL      := $(MODDIRI)/LinkDef.h
WEBGUIDS     := $(call stripsrc,$(MODDIRS)/G__WebGui.cxx)
WEBGUIDO     := $(WEBGUIDS:.cxx=.o)
WEBGUIDH     := $(WEBGUIDS:.cxx=.h)

WEBGUIH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
WEBGUIS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
WEBGUIO      := $(call stripsrc,$(WEBGUIS:.cxx=.o))

WEBGUIDEP    := $(WEBGUIO:.o=.d) $(WEBGUIDO:.o=.d)

WEBGUILIB    := $(LPATH)/libWebGui.$(SOEXT)
WEBGUIMAP    := $(WEBGUILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(WEBGUIH))
ALLLIBS     += $(WEBGUILIB)
ALLMAPS     += $(WEBGUIMAP)

# include all dependency files
INCLUDEFILES += $(WEBGUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(WEBGUIDIRI)/%.h
		cp $< $@

$(WEBGUILIB):   $(WEBGUIO) $(WEBGUIDO) $(ORDER_) $(MAINLIBS) $(WEBGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libWebGui.$(SOEXT) $@ "$(WEBGUIO) $(WEBGUIDO)" \
		   "$(WEBGUILIBEXTRA)"

$(call pcmrule,WEBGUI)
	$(noop)

$(WEBGUIDS):    $(WEBGUIH) $(WEBGUIL) $(ROOTCLINGEXE) $(call pcmdep,WEBGUI)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,WEBGUI) -c $(GQTCXXFLAGS) $(WEBGUIH) $(WEBGUIL)

$(WEBGUIMAP):   $(WEBGUIH) $(WEBGUIL) $(ROOTCLINGEXE) $(call pcmdep,WEBGUI)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(WEBGUIDS) $(call dictModule,WEBGUI) -c $(GQTCXXFLAGS) $(WEBGUIH) $(WEBGUIL)

all-$(MODNAME): $(WEBGUILIB)

clean-$(MODNAME):
		@rm -f $(WEBGUIO) $(WEBGUIDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(WEBGUIDEP) $(WEBGUIDS) $(WEBGUIDH) $(WEBGUILIB) $(WEBGUIMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
