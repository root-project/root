# Module.mk for monalisa module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Andreas Peters, 07/12/2005

MODDIR       := monalisa
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MONALISADIR  := $(MODDIR)
MONALISADIRS := $(MONALISADIR)/src
MONALISADIRI := $(MONALISADIR)/inc

##### libMonaLisa #####
MONALISAL    := $(MODDIRI)/LinkDef.h
MONALISADS   := $(MODDIRS)/G__MonaLisa.cxx
MONALISADO   := $(MONALISADS:.cxx=.o)
MONALISADH   := $(MONALISADS:.cxx=.h)

MONALISAH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MONALISAS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MONALISAO    := $(MONALISAS:.cxx=.o)

MONALISADEP  := $(MONALISAO:.o=.d) $(MONALISADO:.o=.d)

MONALISALIB  := $(LPATH)/libMonaLisa.$(SOEXT)
MONALISAMAP  := $(MONALISALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MONALISAH))
ALLLIBS      += $(MONALISALIB)
ALLMAPS      += $(MONALISAMAP)

# include all dependency files
INCLUDEFILES += $(MONALISADEP)

##### local rules #####
include/%.h:    $(MONALISADIRI)/%.h
		cp $< $@

$(MONALISALIB): $(MONALISAO) $(MONALISADO) $(ORDER_) $(MAINLIBS) $(MONALISALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMonaLisa.$(SOEXT) $@ "$(MONALISAO) $(MONALISADO)" \
		   "$(MONALISALIBEXTRA) $(MONALISALIBDIR) $(MONALISACLILIB) \
		   $(MONALISAWSLIBDIR) $(MONALISAWSCLILIB)"

$(MONALISADS):  $(MONALISAH) $(MONALISAL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MONALISAH) $(MONALISAL)

$(MONALISAMAP): $(RLIBMAP) $(MAKEFILEDEP) $(MONALISAL)
		$(RLIBMAP) -o $(MONALISAMAP) -l $(MONALISALIB) \
		   -d $(MONALISALIBDEPM) -c $(MONALISAL)

all-monalisa:   $(MONALISALIB) $(MONALISAMAP)

clean-monalisa:
		@rm -f $(MONALISAO) $(MONALISADO)

clean::         clean-monalisa

distclean-monalisa: clean-monalisa
		@rm -f $(MONALISADEP) $(MONALISADS) $(MONALISADH) \
		   $(MONALISALIB) $(MONALISAMAP)

distclean::     distclean-monalisa

##### extra rules ######
$(MONALISAO) $(MONALISADO): CXXFLAGS += $(MONALISAINCDIR:%=-I%) $(MONALISAWSINCDIR:%=-I%)
