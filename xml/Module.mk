# Module.mk for xml module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Linev Sergey, Rene Brun 10/05/2004

MODDIR       := xml
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

XMLDIR       := $(MODDIR)
XMLDIRS      := $(XMLDIR)/src
XMLDIRI      := $(XMLDIR)/inc

##### libXMLIO #####
XMLL         := $(MODDIRI)/LinkDef.h
XMLDS        := $(MODDIRS)/G__XML.cxx
XMLDO        := $(XMLDS:.cxx=.o)
XMLDH        := $(XMLDS:.cxx=.h)

XMLH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
XMLS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
XMLO         := $(XMLS:.cxx=.o)

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
include/%.h:    $(XMLDIRI)/%.h
		cp $< $@

$(XMLLIB):      $(XMLO) $(XMLDO) $(ORDER_) $(MAINLIBS) $(XMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLIO.$(SOEXT) $@ "$(XMLO) $(XMLDO)" \
		   "$(XMLLIBEXTRA)"

$(XMLDS):       $(XMLH) $(XMLL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(XMLH) $(XMLL)

$(XMLMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(XMLL)
		$(RLIBMAP) -o $(XMLMAP) -l $(XMLLIB) \
		   -d $(XMLLIBDEPM) -c $(XMLL)

all-xml:        $(XMLLIB) $(XMLMAP)

clean-xml:
		@rm -f $(XMLO) $(XMLDO)

clean::         clean-xml

distclean-xml:  clean-xml
		@rm -f $(XMLDEP) $(XMLDS) $(XMLDH) $(XMLLIB) $(XMLMAP)

distclean::     distclean-xml

