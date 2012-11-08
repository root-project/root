# Module.mk for postscript module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := postscript
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

POSTSCRIPTDIR  := $(MODDIR)
POSTSCRIPTDIRS := $(POSTSCRIPTDIR)/src
POSTSCRIPTDIRI := $(POSTSCRIPTDIR)/inc

##### libPostscript #####
POSTSCRIPTL  := $(MODDIRI)/LinkDef.h
POSTSCRIPTDS := $(call stripsrc,$(MODDIRS)/G__PostScript.cxx)
POSTSCRIPTDO := $(POSTSCRIPTDS:.cxx=.o)
POSTSCRIPTDH := $(POSTSCRIPTDS:.cxx=.h)

POSTSCRIPTH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
POSTSCRIPTS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
POSTSCRIPTO  := $(call stripsrc,$(POSTSCRIPTS:.cxx=.o))

POSTSCRIPTDEP := $(POSTSCRIPTO:.o=.d) $(POSTSCRIPTDO:.o=.d)

POSTSCRIPTLIB := $(LPATH)/libPostscript.$(SOEXT)
POSTSCRIPTMAP := $(POSTSCRIPTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(POSTSCRIPTH))
ALLLIBS       += $(POSTSCRIPTLIB)
ALLMAPS       += $(POSTSCRIPTMAP)

# include all dependency files
INCLUDEFILES += $(POSTSCRIPTDEP)

ifneq ($(BUILTINZLIB),yes)
POSTSCRIPTLIBEXTRA += $(ZLIBLIBDIR) $(ZLIBCLILIB)
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(POSTSCRIPTDIRI)/%.h
		cp $< $@

$(POSTSCRIPTLIB): $(POSTSCRIPTO) $(POSTSCRIPTDO) $(MATHTEXTLIBDEP) $(FREETYPEDEP) \
                     $(ORDER_) $(MAINLIBS) $(POSTSCRIPTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPostscript.$(SOEXT) $@ \
		   "$(POSTSCRIPTO) $(POSTSCRIPTDO)" \
		   "$(POSTSCRIPTLIBEXTRA) $(MATHTEXTLIB) $(FREETYPELDFLAGS) $(FREETYPELIB)"

$(POSTSCRIPTDS): $(POSTSCRIPTH) $(POSTSCRIPTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(POSTSCRIPTH) $(POSTSCRIPTL)

$(POSTSCRIPTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(POSTSCRIPTL)
		$(RLIBMAP) -o $@ -l $(POSTSCRIPTLIB) \
		   -d $(POSTSCRIPTLIBDEPM) -c $(POSTSCRIPTL)

all-$(MODNAME): $(POSTSCRIPTLIB) $(POSTSCRIPTMAP)

clean-$(MODNAME):
		@rm -f $(POSTSCRIPTO) $(POSTSCRIPTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(POSTSCRIPTDEP) $(POSTSCRIPTDS) $(POSTSCRIPTDH) \
		   $(POSTSCRIPTLIB) $(POSTSCRIPTMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
