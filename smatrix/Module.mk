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
#SMATRIXLINC :=
SMATRIXDS   := $(MODDIRS)/G__Smatrix.cxx
SMATRIXDO   := $(SMATRIXDS:.cxx=.o)
SMATRIXDH   := $(SMATRIXDS:.cxx=.h)

SMATRIXDH1  :=  $(MODDIRI)/Math/SMatrix.h $(MODDIRI)/Math/SVector.h



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
		   mkdir include/Math;                 \
		fi)
		cp $< $@

include/Math/%.icc: $(SMATRIXDIRI)/%.icc
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir include/Math;                 \
		fi)
		cp $< $@

$(SMATRIXLIB): $(SMATRIXO) $(SMATRIXDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libSmatrix.$(SOEXT) $@     \
		   "$(SMATRIXO) $(SMATRIXDO)"             \
		   "$(SMATRIXLIBEXTRA)"


$(SMATRIXDS):  $(SMATRIXDH1) $(SMATRIXL) $(SMATRIXLINC) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@echo "for files $(SMATRIXDH1)"
		$(ROOTCINTTMP) -f $@ -c $(SMATRIXDH1) $(SMATRIXL)
#		python reflex/python/genreflex/genreflex.py $(SMATRIXDIRS)/Dict.h -I$(SMATRIXDIRI) --selection_file=$(SMATRIXDIRS)/Selection.xml -o $(SMATRIXDIRS)/G__Smatrix.cxx


$(SMATRIXDO):  $(SMATRIXDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(SMATRIXDIRI) -o $@ -c $<

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

distclean::     distclean-smatrix
