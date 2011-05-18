# Module.mk for matrix module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := matrix
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATRIXDIR    := $(MODDIR)
MATRIXDIRS   := $(MATRIXDIR)/src
MATRIXDIRI   := $(MATRIXDIR)/inc

##### libMatrix #####
MATRIXL      := $(MODDIRI)/LinkDef.h
MATRIXDS     := $(call stripsrc,$(MODDIRS)/G__Matrix.cxx)
MATRIXDO     := $(MATRIXDS:.cxx=.o)
MATRIXDH     := $(MATRIXDS:.cxx=.h)

MATRIXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATRIXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATRIXO      := $(call stripsrc,$(MATRIXS:.cxx=.o))

MATRIXDEP    := $(MATRIXO:.o=.d) $(MATRIXDO:.o=.d)

MATRIXLIB    := $(LPATH)/libMatrix.$(SOEXT)
MATRIXMAP    := $(MATRIXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MATRIXH))
ALLLIBS     += $(MATRIXLIB)
ALLMAPS     += $(MATRIXMAP)

# include all dependency files
INCLUDEFILES += $(MATRIXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MATRIXDIRI)/%.h
		cp $< $@

$(MATRIXLIB):   $(MATRIXO) $(MATRIXDO) $(ORDER_) $(MAINLIBS) $(MATRIXLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMatrix.$(SOEXT) $@ "$(MATRIXO) $(MATRIXDO)" \
		   "$(MATRIXLIBEXTRA)"

$(MATRIXDS):    $(MATRIXH) $(MATRIXL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MATRIXH) $(MATRIXL)

$(MATRIXMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(MATRIXL)
		$(RLIBMAP) -o $@ -l $(MATRIXLIB) \
		   -d $(MATRIXLIBDEPM) -c $(MATRIXL)

all-$(MODNAME): $(MATRIXLIB) $(MATRIXMAP)

clean-$(MODNAME):
		@rm -f $(MATRIXO) $(MATRIXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MATRIXDEP) $(MATRIXDS) $(MATRIXDH) $(MATRIXLIB) $(MATRIXMAP)

distclean::     distclean-$(MODNAME)
