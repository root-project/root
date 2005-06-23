# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := reflex
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

REFLEXDIR    := $(MODDIR)
REFLEXDIRS   := $(REFLEXDIR)/src
REFLEXDIRI   := $(REFLEXDIR)/inc

##### libReflex #####
REFLEXAH     := $(wildcard $(MODDIRI)/Reflex/*.h)
REFLEXBH     := $(wildcard $(MODDIRI)/Reflex/Builder/*.h)
REFLEXH      := $(REFLEXAH) $(REFLEXBH)
REFLEXS      := $(wildcard $(MODDIRS)/*.cxx)
REFLEXO      := $(REFLEXS:.cxx=.o)

REFLEXDEP    := $(REFLEXO:.o=.d)

REFLEXLIB    := $(LPATH)/libReflex.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Reflex/%.h,include/Reflex/%.h,$(REFLEXH))
ALLLIBS      += $(REFLEXLIB)

# include all dependency files
INCLUDEFILES += $(REFLEXDEP)

##### local rules #####
include/Reflex/%.h: $(REFLEXDIRI)/Reflex/%.h
		@(if [ ! -d "include/Reflex" ]; then    \
		   mkdir -p include/Reflex/Builder;     \
		fi)
		cp $< $@

$(REFLEXLIB):   $(REFLEXO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflex.$(SOEXT) $@ "$(REFLEXO)" \
		"$(REFLEXLIBEXTRA)"

all-reflex:     $(REFLEXLIB)

map-reflex:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(REFLEXLIB) \
		-d $(REFLEXLIBDEP) -c $(REFLEXL)

map::           map-reflex

clean-reflex:
		@rm -f $(REFLEXO)

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB)

distclean::     distclean-reflex

