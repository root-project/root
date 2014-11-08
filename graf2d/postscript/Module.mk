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
POSTSCRIPTDS := $(call stripsrc,$(MODDIRS)/G__Postscript.cxx)
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

$(call pcmrule,POSTSCRIPT)
	$(noop)

$(POSTSCRIPTDS): $(POSTSCRIPTH) $(POSTSCRIPTL) $(ROOTCLINGEXE) $(call pcmdep,POSTSCRIPT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,POSTSCRIPT) -c -writeEmptyRootPCM $(POSTSCRIPTH) $(POSTSCRIPTL)

$(POSTSCRIPTMAP): $(POSTSCRIPTH) $(POSTSCRIPTL) $(ROOTCLINGEXE) $(call pcmdep,POSTSCRIPT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(POSTSCRIPTDS) $(call dictModule,POSTSCRIPT) -c $(POSTSCRIPTH) $(POSTSCRIPTL)

all-$(MODNAME): $(POSTSCRIPTLIB)
clean-$(MODNAME):
		@rm -f $(POSTSCRIPTO) $(POSTSCRIPTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(POSTSCRIPTDEP) $(POSTSCRIPTDS) $(POSTSCRIPTDH) \
		   $(POSTSCRIPTLIB) $(POSTSCRIPTMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
