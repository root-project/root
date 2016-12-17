# Module.mk for xmlparser module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Authors: Jose Lo

MODNAME       := xmlparser
MODDIR        := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

XMLPARSERDIR  := $(MODDIR)
XMLPARSERDIRS := $(XMLPARSERDIR)/src
XMLPARSERDIRI := $(XMLPARSERDIR)/inc

##### libXMLParser #####
XMLPARSERL    := $(MODDIRI)/LinkDef.h
XMLPARSERDS   := $(call stripsrc,$(MODDIRS)/G__XMLParser.cxx)
XMLPARSERDO   := $(XMLPARSERDS:.cxx=.o)
XMLPARSERDH   := $(XMLPARSERDS:.cxx=.h)

XMLPARSERH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
XMLPARSERS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
XMLPARSERO    := $(call stripsrc,$(XMLPARSERS:.cxx=.o))

XMLPARSERDEP  := $(XMLPARSERO:.o=.d) $(XMLPARSERDO:.o=.d)

XMLPARSERLIB  := $(LPATH)/libXMLParser.$(SOEXT)
XMLPARSERMAP  := $(XMLPARSERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
XMLPARSERH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(XMLPARSERH))
ALLHDRS      += $(XMLPARSERH_REL)
ALLLIBS      += $(XMLPARSERLIB)
ALLMAPS      += $(XMLPARSERMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(XMLPARSERH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Io_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(XMLPARSERLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(XMLPARSERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(XMLPARSERDIRI)/%.h
		cp $< $@

$(XMLPARSERLIB): $(XMLPARSERO) $(XMLPARSERDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXMLParser.$(SOEXT) $@ \
		   "$(XMLPARSERO) $(XMLPARSERDO)" \
		   "$(XMLLIBDIR) $(XMLCLILIB)"

$(call pcmrule,XMLPARSER)
	$(noop)

$(XMLPARSERDS): $(XMLPARSERH) $(XMLPARSERL) $(ROOTCLINGEXE) $(call pcmdep,XMLPARSER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,XMLPARSER) -c $(XMLPARSERH) $(XMLPARSERL)

$(XMLPARSERMAP): $(XMLPARSERH) $(XMLPARSERL) $(ROOTCLINGEXE) $(call pcmdep,XMLPARSER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(XMLPARSERDS) $(call dictModule,XMLPARSER) -c $(XMLPARSERH) $(XMLPARSERL)

all-$(MODNAME): $(XMLPARSERLIB)
clean-$(MODNAME):
		@rm -f $(XMLPARSERO) $(XMLPARSERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(XMLPARSERDEP) $(XMLPARSERDS) $(XMLPARSERDH) \
		   $(XMLPARSERLIB) $(XMLPARSERMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(XMLPARSERO): CXXFLAGS += $(XMLINCDIR:%=-I%)
