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
XMLPARSERMAP  := $(XMLPARSERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(XMLPARSERH))
ALLLIBS      += $(XMLPARSERLIB)
ALLMAPS      += $(XMLPARSERMAP)

# include all dependency files
INCLUDEFILES += $(XMLPARSERDEP)

##### local rules #####
include/%.h:    $(XMLPARSERDIRI)/%.h
		cp $< $@

$(XMLPARSERLIB): $(XMLPARSERO) $(XMLPARSERDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLParser.$(SOEXT) $@ \
		   "$(XMLPARSERO) $(XMLPARSERDO)" \
		   "$(XMLLIBDIR) $(XMLCLILIB)"

$(XMLPARSERDS): $(XMLPARSERH) $(XMLPARSERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(XMLPARSERH) $(XMLPARSERL)

$(XMLPARSERMAP): $(RLIBMAP) $(MAKEFILEDEP) $(XMLPARSERL)
		$(RLIBMAP) -o $(XMLPARSERMAP) -l $(XMLPARSERLIB) \
		   -d $(XMLPARSERLIBDEPM) -c $(XMLPARSERL)

all-xmlparser:  $(XMLPARSERLIB) $(XMLPARSERMAP)

clean-xmlparser:
		@rm -f $(XMLPARSERO) $(XMLPARSERDO)

clean::         clean-xmlparser

distclean-xmlparser: clean-xmlparser
		@rm -f $(XMLPARSERDEP) $(XMLPARSERDS) $(XMLPARSERDH) \
		   $(XMLPARSERLIB) $(XMLPARSERMAP)

distclean::     distclean-xmlparser

##### extra rules ######
$(XMLPARSERO): CXXFLAGS += $(XMLINCDIR:%=-I%)
