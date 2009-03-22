# Module.mk for meta module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := meta
MODDIR       := core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

METADIR      := $(MODDIR)
METADIRS     := $(METADIR)/src
METADIRI     := $(METADIR)/inc

##### libMeta (part of libCore) #####
METAL        := $(MODDIRI)/LinkDef.h
METADS       := $(MODDIRS)/G__Meta.cxx
METADO       := $(METADS:.cxx=.o)
METADH       := $(METADS:.cxx=.h)

METAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAH        := $(filter-out $(MODDIRI)/TCint.h,$(filter-out $(MODDIRI)/TCint_7.h,$(METAH)))
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
METAS        := $(filter-out $(MODDIRS)/TCint.cxx,$(filter-out $(MODDIRS)/TCint_7.cxx,$(METAS)))
METAO        := $(METAS:.cxx=.o)

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

##### libMetaTCint #####
METATCINTL   := $(MODDIRI)/LinkDef_TCint.h
METATCINTDS  := $(MODDIRS)/G__TCint.cxx
METATCINTDO  := $(METATCINTDS:.cxx=.o)
METATCINTDH  := $(METATCINTDS:.cxx=.h)
METATCINTH   := $(MODDIRI)/TCint.h
METATCINTS   := $(MODDIRS)/TCint.cxx
METATCINTO   := $(METATCINTS:.cxx=.o)
METATCINTDEP := $(METATCINTO:.o=.d) $(METATCINTDO:.o=.d)
METATCINTLIB := $(LPATH)/libMetaTCint.$(SOEXT)
METATCINTMAP := $(METATCINTLIB:.$(SOEXT)=.rootmap)

##### libMetaTCint_7 #####
ifneq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
BUILDBOTHCINT := yes
endif
ifneq ($(BUILDBOTHCINT),)
METATCINT7L   := $(MODDIRI)/LinkDef_TCint.h # intentionally identical to libMetaTCint!
METATCINT7DS  := $(MODDIRS)/G__TCint_7.cxx
METATCINT7DO  := $(METATCINT7DS:.cxx=.o)
METATCINT7DH  := $(METATCINT7DS:.cxx=.h)
METATCINT7H   := $(MODDIRI)/TCint_7.h
METATCINT7S   := $(MODDIRS)/TCint_7.cxx
METATCINT7O   := $(METATCINT7S:.cxx=.o)
METATCINT7DEP := $(METATCINT7O:.o=.d) $(METATCINT7DO:.o=.d)
METATCINT7LIB := $(LPATH)/libMetaTCint_7.$(SOEXT)
METATCINT7MAP := $(METATCINT7LIB:.$(SOEXT)=.rootmap)

ALLLIBS     += $(METATCINTLIB) $(METATCINT7LIB)
ALLMAPS     += $(METATCINTMAP) $(METATCINT7MAP)
else
# just add TCint to libCore
METAO       += $(METATCINTO) $(METATCINTDO)
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH) $(METATCINTH) $(METATCINT7H))

# include all dependency files
INCLUDEFILES += $(METADEP) $(METATCINTDEP) $(METATCINT7DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METADIRI)/%.h
		cp $< $@

ifneq ($(BUILDBOTHCINT),)
$(METATCINT7H): $(METATCINTH)
		cp $< $@

$(METATCINT7S): $(METATCINTS)
		cp $< $@
endif

$(METATCINTLIB): $(METATCINTO) $(METATCINTDO) $(ORDER_)  $(CINTLIB) $(CORELIB) \
                 $(METATCINTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMetaTCint.$(SOEXT) $@ "$(METATCINTO) $(METATCINTDO)" \
		   "$(METATCINTLIBEXTRA)"

ifneq ($(BUILDBOTHCINT),)
$(METATCINT7LIB): $(METATCINT7O) $(METATCINT7DO) $(ORDER_) $(CINT7LIB)  $(CINTLIB) $(CORELIB) \
                  $(METATCINT7LIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMetaTCint_7.$(SOEXT) $@ \
		   "$(METATCINT7O) $(METATCINT7DO)" \
		   "$(METATCINT7LIBEXTRA)"
endif

$(METADS):      $(METAH) $(METAL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAH) $(METAL)

$(METATCINTDS): $(METATCINTH) $(METATCINTL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METATCINTH) $(METATCINTL)

ifneq ($(BUILDBOTHCINT),)
$(METATCINT7DS): $(METATCINT7H) $(METATCINT7L) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METATCINT7H) $(METATCINT7L)
endif

$(METATCINTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(METATCINTL)
		$(RLIBMAP) -o $(METATCINTMAP) -l $(METATCINTLIB) \
		   -d $(METATCINTLIBDEPM) -c $(METATCINTL)

ifneq ($(BUILDBOTHCINT),)
$(METATCINT7MAP): $(RLIBMAP) $(MAKEFILEDEP) $(METATCINT7L)
		$(RLIBMAP) -o $(METATCINT7MAP) -l $(METATCINT7LIB) \
		   -d $(METATCINT7LIBDEPM) -c $(METATCINT7L)
endif

all-$(MODNAME): $(METAO) $(METADO) $(METATCINTLIB) $(METATCINT7LIB) \
                $(METATCINTMAP) $(METATCINT7MAP)

clean-$(MODNAME):
		@rm -f $(METAO) $(METADO) $(METATCINTO) $(METATCINTDO) \
		   $(METATCINT7O) $(METATCINT7DO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(METADS) $(METADH) \
		  $(METATCINTDEP) $(METATCINTDS) $(METATCINTDH) \
		  $(METATCINT7DEP) $(METATCINT7DS) $(METATCINT7DH) \
		  $(METATCINTLIB) $(METATCINT7LIB) \
		  $(METATCINTMAP) $(METATCINT7MAP) \
		  $(METATCINT7H) $(METATCINT7S)

distclean::     distclean-$(MODNAME)

ifneq ($(BUILDBOTHCINT),)
$(METATCINT7O): CXXFLAGS += -DR__BUILDING_CINT7
else
ifeq ($(BUILDCINT7),yes)
$(METATCINTO): CXXFLAGS += -DR__BUILDING_ONLYCINT7
endif
endif
