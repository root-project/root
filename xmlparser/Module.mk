# Module.mk for xmlparser module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Jose Lo

MODDIR        := xmlparser
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

XMLPARSERDIR  := $(MODDIR)
XMLPARSERDIRS := $(XMLPARSERDIR)/src
XMLPARSERDIRI := $(XMLPARSERDIR)/inc

##### libXMLParser #####
XMLPARSERL    := $(MODDIRI)/LinkDef.h
XMLPARSERDS   := $(MODDIRS)/G__XMLParser.cxx
XMLPARSERDO   := $(XMLPARSERDS:.cxx=.o)
XMLPARSERDH   := $(XMLPARSERDS:.cxx=.h)

XMLPARSERH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
XMLPARSERS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
XMLPARSERO    := $(XMLPARSERS:.cxx=.o)

XMLPARSERDEP  := $(XMLPARSERO:.o=.d) $(XMLPARSERDO:.o=.d)

XMLPARSERLIB  := $(LPATH)/libXMLParser.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(XMLPARSERH))
ALLLIBS      += $(XMLPARSERLIB)

# include all dependency files
INCLUDEFILES += $(XMLPARSERDEP)

##### local rules #####
include/%.h:    $(XMLPARSERDIRI)/%.h
		cp $< $@

$(XMLPARSERLIB): $(XMLPARSERO) $(XMLPARSERDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLParser.$(SOEXT) $@ \
		   "$(XMLPARSERO) $(XMLPARSERDO)" \
		   "$(XMLLIBDIR) $(XMLCLILIB)"

$(XMLPARSERDS): $(XMLPARSERH) $(XMLPARSERL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(XMLPARSERH) $(XMLPARSERL)

$(XMLPARSERDO): $(XMLPARSERDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-xmlparser:  $(XMLPARSERLIB)

map-xmlparser:  $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(XMLPARSERLIB) \
		   -d $(XMLPARSERLIBDEP) -c $(XMLPARSERL)

map::           map-xmlparser

clean-xmlparser:
		@rm -f $(XMLPARSERO) $(XMLPARSERDO)

clean::         clean-xmlparser

distclean-xmlparser: clean-xmlparser
		@rm -f $(XMLPARSERDEP) $(XMLPARSERDS) $(XMLPARSERDH) \
		   $(XMLPARSERLIB)

distclean::     distclean-xmlparser

##### extra rules ######
$(XMLPARSERO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(XMLINCDIR:%=-I%) -o $@ -c $<
