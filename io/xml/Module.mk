# Module.mk for xml module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Linev Sergey, Rene Brun 10/05/2004

MODNAME      := xml
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

XMLDIR       := $(MODDIR)
XMLDIRS      := $(XMLDIR)/src
XMLDIRI      := $(XMLDIR)/inc

##### libXMLIO #####
XMLL         := $(MODDIRI)/LinkDef.h
XMLDS        := $(call stripsrc,$(MODDIRS)/G__XMLIO.cxx)
XMLDO        := $(XMLDS:.cxx=.o)
XMLDH        := $(XMLDS:.cxx=.h)

XMLH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
XMLS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
XMLO         := $(call stripsrc,$(XMLS:.cxx=.o))

XMLDEP       := $(XMLO:.o=.d) $(XMLDO:.o=.d)

XMLLIB       := $(LPATH)/libXMLIO.$(SOEXT)
XMLMAP       := $(XMLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(XMLH))
ALLLIBS     += $(XMLLIB)
ALLMAPS     += $(XMLMAP)

# include all dependency files
INCLUDEFILES += $(XMLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(XMLDIRI)/%.h
		cp $< $@

$(XMLLIB):      $(XMLO) $(XMLDO) $(ORDER_) $(MAINLIBS) $(XMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLIO.$(SOEXT) $@ "$(XMLO) $(XMLDO)" \
		   "$(XMLLIBEXTRA)"

$(call pcmrule,XML)
	$(noop)

$(XMLDS):       $(XMLH) $(XMLL) $(ROOTCLINGEXE) $(call pcmdep,XML)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,XML) -c $(XMLH) $(XMLL)

$(XMLMAP):      $(XMLH) $(XMLL) $(ROOTCLINGEXE) $(call pcmdep,XML)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(XMLDS) $(call dictModule,XML) -c $(XMLH) $(XMLL)

all-$(MODNAME): $(XMLLIB)

clean-$(MODNAME):
		@rm -f $(XMLO) $(XMLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(XMLDEP) $(XMLDS) $(XMLDH) $(XMLLIB) $(XMLMAP)

distclean::     distclean-$(MODNAME)
