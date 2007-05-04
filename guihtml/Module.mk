# Module.mk for guibuilder module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Valeriy Onuchin, 24/4/2007

MODDIR       := guihtml
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GUIHTMLDIR    := $(MODDIR)
GUIHTMLDIRS   := $(GUIHTMLDIR)/src
GUIHTMLDIRI   := $(GUIHTMLDIR)/inc

##### libGuiHtml #####
GUIHTMLL      := $(MODDIRI)/LinkDef.h
GUIHTMLDS     := $(MODDIRS)/G__GuiHtml.cxx
GUIHTMLDO     := $(GUIHTMLDS:.cxx=.o)
GUIHTMLDH     := $(GUIHTMLDS:.cxx=.h)

GUIHTMLH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GUIHTMLS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GUIHTMLO      := $(GUIHTMLS:.cxx=.o)

GUIHTMLDEP    := $(GUIHTMLO:.o=.d) $(GUIHTMLDO:.o=.d)

GUIHTMLLIB    := $(LPATH)/libGuiHtml.$(SOEXT)
GUIHTMLMAP    := $(GUIHTMLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GUIHTMLH))
ALLLIBS     += $(GUIHTMLLIB)
ALLMAPS     += $(GUIHTMLMAP)

# include all dependency files
INCLUDEFILES += $(GUIHTMLDEP)

##### local rules #####
include/%.h:    $(GUIHTMLDIRI)/%.h
		cp $< $@

$(GUIHTMLLIB):   $(GUIHTMLO) $(GUIHTMLDO) $(ORDER_) $(MAINLIBS) $(GUIHTMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGuiHtml.$(SOEXT) $@ "$(GUIHTMLO) $(GUIHTMLDO)" \
		   "$(GUIHTMLLIBEXTRA)"

$(GUIHTMLDS):    $(GUIHTMLH) $(GUIHTMLL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GUIHTMLH) $(GUIHTMLL)

$(GUIHTMLMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(GUIHTMLL)
		$(RLIBMAP) -o $(GUIHTMLMAP) -l $(GUIHTMLLIB) \
		   -d $(GUIHTMLLIBDEPM) -c $(GUIHTMLL)

all-guihtml: $(GUIHTMLLIB) $(GUIHTMLMAP)

clean-guihtml:
		@rm -f $(GUIHTMLO) $(GUIHTMLDO)

clean::         clean-guihtml

distclean-guihtml: clean-guihtml
		@rm -f $(GUIHTMLDEP) $(GUIHTMLDS) $(GUIHTMLDH) $(GUIHTMLLIB) $(GUIHTMLMAP)

distclean::     distclean-guihtml
