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

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(XMLH))
ALLLIBS     += $(XMLLIB)

# include all dependency files
INCLUDEFILES += $(XMLDEP)

##### local rules #####
include/%.h:    $(XMLDIRI)/%.h
		cp $< $@

$(XMLLIB):      $(XMLO) $(XMLDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLIO.$(SOEXT) $@ "$(XMLO) $(XMLDO)"

$(XMLDS):       $(XMLH) $(XMLL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(XMLH) $(XMLL)

$(XMLDO):       $(XMLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-xml:        $(XMLLIB)

map-xml:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(XMLLIB) -d $(XMLLIBDEP) -c $(XMLL)

map::           map-xml

clean-xml:
		@rm -f $(XMLO) $(XMLDO)

clean::         clean-xml

distclean-xml:  clean-xml
		@rm -f $(XMLDEP) $(XMLDS) $(XMLDH) $(XMLLIB)

distclean::     distclean-xml

