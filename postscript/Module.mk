# Module.mk for postscript module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := postscript
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

POSTSCRIPTDIR  := $(MODDIR)
POSTSCRIPTDIRS := $(POSTSCRIPTDIR)/src
POSTSCRIPTDIRI := $(POSTSCRIPTDIR)/inc

##### libTree #####
POSTSCRIPTL  := $(MODDIRI)/LinkDef.h
POSTSCRIPTDS := $(MODDIRS)/G__PostScript.cxx
POSTSCRIPTDO := $(POSTSCRIPTDS:.cxx=.o)
POSTSCRIPTDH := $(POSTSCRIPTDS:.cxx=.h)

POSTSCRIPTH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
POSTSCRIPTS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
POSTSCRIPTO  := $(POSTSCRIPTS:.cxx=.o)

POSTSCRIPTDEP := $(POSTSCRIPTO:.o=.d) $(POSTSCRIPTDO:.o=.d)

POSTSCRIPTLIB := $(LPATH)/libPostscript.$(SOEXT)
POSTSCRIPTMAP := $(POSTSCRIPTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(POSTSCRIPTH))
ALLLIBS       += $(POSTSCRIPTLIB)
ALLMAPS       += $(POSTSCRIPTMAP)

# include all dependency files
INCLUDEFILES += $(POSTSCRIPTDEP)

##### local rules #####
include/%.h:    $(POSTSCRIPTDIRI)/%.h
		cp $< $@

$(POSTSCRIPTLIB): $(POSTSCRIPTO) $(POSTSCRIPTDO) $(ORDER_) $(MAINLIBS) $(POSTSCRIPTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPostscript.$(SOEXT) $@ \
		   "$(POSTSCRIPTO) $(POSTSCRIPTDO)" \
		   "$(POSTSCRIPTLIBEXTRA)"

$(POSTSCRIPTDS): $(POSTSCRIPTH) $(POSTSCRIPTL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(POSTSCRIPTH) $(POSTSCRIPTL)

$(POSTSCRIPTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(POSTSCRIPTL)
		$(RLIBMAP) -o $(POSTSCRIPTMAP) -l $(POSTSCRIPTLIB) \
		   -d $(POSTSCRIPTLIBDEPM) -c $(POSTSCRIPTL)

all-postscript: $(POSTSCRIPTLIB) $(POSTSCRIPTMAP)

clean-postscript:
		@rm -f $(POSTSCRIPTO) $(POSTSCRIPTDO)

clean::         clean-postscript

distclean-postscript: clean-postscript
		@rm -f $(POSTSCRIPTDEP) $(POSTSCRIPTDS) $(POSTSCRIPTDH) \
		   $(POSTSCRIPTLIB) $(POSTSCRIPTMAP)

distclean::     distclean-postscript

##### extra rules ######
ifeq ($(ARCH),alphacxx6)
$(POSTSCRIPTDIRS)/TPostScript.o: OPT = $(NOOPT)
endif
