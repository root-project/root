# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := cintex
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CINTEXDIR    := $(MODDIR)
CINTEXDIRS   := $(CINTEXDIR)/src
CINTEXDIRI   := $(CINTEXDIR)/inc

##### libCintex #####
CINTEXH      := $(wildcard $(MODDIRI)/Cintex/*.h)
CINTEXS      := $(wildcard $(MODDIRS)/*.cxx)
CINTEXO      := $(CINTEXS:.cxx=.o)

CINTEXDEP    := $(CINTEXO:.o=.d)

CINTEXLIB    := $(LPATH)/libCintex.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Cintex/%.h,include/Cintex/%.h,$(CINTEXH))
ALLLIBS      += $(CINTEXLIB)

# include all dependency files
INCLUDEFILES += $(CINTEXDEP)

##### local rules #####
include/Cintex/%.h: $(CINTEXDIRI)/Cintex/%.h
		@(if [ ! -d "include/Cintex" ]; then    \
		   mkdir -p include/Cintex;             \
		fi)
		cp $< $@

$(CINTEXLIB):   $(CINTEXO) $(MAINLIBS) $(CINTEXLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libCintex.$(SOEXT) $@ "$(CINTEXO)" \
		"$(CINTEXLIBEXTRA)"

all-cintex:     $(CINTEXLIB)

map-cintex:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(CINTEXLIB) \
		-d $(CINTEXLIBDEP) -c $(CINTEXL)

map::           map-cintex

clean-cintex:
		@rm -f $(CINTEXO)

clean::         clean-cintex

distclean-cintex: clean-cintex
		@rm -f $(CINTEXDEP) $(CINTEXLIB)
		@rm -rf include/Cintex

distclean::     distclean-cintex


