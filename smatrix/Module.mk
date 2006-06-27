# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODDIR       := smatrix
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SMATRIXDIR  := $(MODDIR)
SMATRIXDIRS := $(SMATRIXDIR)/src
SMATRIXDIRI := $(SMATRIXDIR)/inc/Math

##### libSmatrix #####
SMATRIXL    := $(MODDIRI)/LinkDef.h
SMATRIXL32  := $(MODDIRI)/LinkDefD32.h
#SMATRIXLINC :=
SMATRIXDS   := $(MODDIRS)/G__Smatrix.cxx
SMATRIXDS32 := $(MODDIRS)/G__Smatrix32.cxx
SMATRIXDO   := $(SMATRIXDS:.cxx=.o)
SMATRIXDO32 := $(SMATRIXDS32:.cxx=.o)
SMATRIXDH   := $(SMATRIXDS:.cxx=.h)

SMATRIXDH1  :=  $(MODDIRI)/Math/SMatrix.h \
		$(MODDIRI)/Math/SVector.h \
		$(MODDIRI)/Math/SMatrixDfwd.h \
		$(MODDIRI)/Math/SMatrixFfwd.h \
		$(MODDIRI)/Math/SMatrixD32fwd.h 



SMATRIXH1   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))
SMATRIXH2   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.icc))
SMATRIXH    := $(SMATRIXH1) $(SMATRIXH2)
SMATRIXS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SMATRIXO    := $(SMATRIXS:.cxx=.o)

SMATRIXDEP  := $(SMATRIXO:.o=.d)  $(SMATRIXDO:.o=.d)

SMATRIXLIB  := $(LPATH)/libSmatrix.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(SMATRIXH1))
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.icc,include/Math/%.icc,$(SMATRIXH2))
ALLLIBS      += $(SMATRIXLIB)

# include all dependency files
INCLUDEFILES += $(SMATRIXDEP)

##### local rules #####
include/Math/%.h: $(SMATRIXDIRI)/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

include/Math/%.icc: $(SMATRIXDIRI)/%.icc
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

$(SMATRIXLIB): $(SMATRIXO) $(SMATRIXDO) $(SMATRIXDO32) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libSmatrix.$(SOEXT) $@     \
		   "$(SMATRIXO) $(SMATRIXDO) $(SMATRIXDO32)"             \
		   "$(SMATRIXLIBEXTRA)"

$(SMATRIXDS):  $(SMATRIXDH1) $(SMATRIXL) $(SMATRIXLINC) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		@echo "for files $(SMATRIXDH1)"
		$(ROOTCINTTMP) -f $@ -c $(SMATRIXDH1) $(SMATRIXL)
#		python reflex/python/genreflex/genreflex.py $(SMATRIXDIRS)/Dict.h -I$(SMATRIXDIRI) --selection_file=$(SMATRIXDIRS)/Selection.xml -o $(SMATRIXDIRS)/G__Smatrix.cxx

$(SMATRIXDS32): $(SMATRIXDH1) $(SMATRIXL32) $(SMATRIXLINC) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		@echo "for files $(SMATRIXDH1)"
		$(ROOTCINTTMP) -f $@ -c $(SMATRIXDH1) $(SMATRIXL32)


ifneq ($(ICC_MAJOR),)
# silence warning messages about subscripts being out of range
$(SMATRIXDO): CXXFLAGS += -wd175 -I$(SMATRIXDIRI)
else
$(SMATRIXDO): CXXFLAGS += -I$(SMATRIXDIRI)
endif

all-smatrix:   $(SMATRIXLIB)

map-smatrix:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(SMATRIXLIB) \
		   -d $(SMATRIXLIBDEP) -c $(SMATRIXL) $(SMATRIXLINC)

map::           map-smatrix

clean-smatrix:
		@rm -f $(SMATRIXO) $(SMATRIXDO)

clean::         clean-smatrix

distclean-smatrix: clean-smatrix
		@rm -f $(SMATRIXDEP) $(SMATRIXDS) $(SMATRIXDH) $(SMATRIXLIB)
		@rm -rf include/Math
		-@cd $(SMATRIXDIR)/test && $(MAKE) distclean

distclean::     distclean-smatrix

test-smatrix: 	all-smatrix
		@cd $(SMATRIXDIR)/test && $(MAKE)
