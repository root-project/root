# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODNAME      := smatrix
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SMATRIXDIR  := $(MODDIR)
SMATRIXDIRS := $(SMATRIXDIR)/src
SMATRIXDIRI := $(SMATRIXDIR)/inc/Math
SMATRIXDIRT := $(call stripsrc,$(SMATRIXDIR)/test)

##### libSmatrix #####
SMATRIXL    := $(MODDIRI)/LinkDef.h
SMATRIXL32  := $(MODDIRI)/LinkDefD32.h
#SMATRIXLINC :=
SMATRIXDS   := $(call stripsrc,$(MODDIRS)/G__Smatrix.cxx)
SMATRIXDS32 := $(call stripsrc,$(MODDIRS)/G__Smatrix32.cxx)
SMATRIXDO   := $(SMATRIXDS:.cxx=.o)
SMATRIXDO32 := $(SMATRIXDS32:.cxx=.o)
SMATRIXDH   := $(SMATRIXDS:.cxx=.h)
SMATRIXDH32 := $(SMATRIXDS32:.cxx=.h)

SMATRIXDH1  :=  $(MODDIRI)/Math/SMatrix.h \
		$(MODDIRI)/Math/SVector.h \
		$(MODDIRI)/Math/SMatrixDfwd.h \
		$(MODDIRI)/Math/SMatrixFfwd.h 
#		$(MODDIRI)/Math/SMatrixD32fwd.h



SMATRIXH1   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))
SMATRIXH2   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.icc))
SMATRIXH    := $(SMATRIXH1) $(SMATRIXH2)
SMATRIXS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SMATRIXO    := $(call stripsrc,$(SMATRIXS:.cxx=.o))

SMATRIXDEP  := $(SMATRIXO:.o=.d)  $(SMATRIXDO:.o=.d) $(SMATRIXDO32:.o=.d)

SMATRIXLIB  := $(LPATH)/libSmatrix.$(SOEXT)
SMATRIXMAP  := $(SMATRIXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(SMATRIXH1))
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.icc,include/Math/%.icc,$(SMATRIXH2))
ALLLIBS      += $(SMATRIXLIB)
ALLMAPS      += $(SMATRIXMAP)

# include all dependency files
INCLUDEFILES += $(SMATRIXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

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
		   "$(SMATRIXO) $(SMATRIXDO) $(SMATRIXDO32)" \
		   "$(SMATRIXLIBEXTRA)"

$(SMATRIXDS):  $(SMATRIXDH1) $(SMATRIXL) $(SMATRIXLINC) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(MAKEDIR)
		$(ROOTCINTTMP) -f $@ -c $(SMATRIXDH1) $(SMATRIXL)

$(SMATRIXDS32): $(SMATRIXDH1) $(SMATRIXL32) $(SMATRIXLINC) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(MAKEDIR)
		$(ROOTCINTTMP) -f $@ -c $(SMATRIXDH1) $(SMATRIXL32)

$(SMATRIXMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(SMATRIXL) $(SMATRIXLINC)
		$(RLIBMAP) -o $@ -l $(SMATRIXLIB) \
		   -d $(SMATRIXLIBDEPM) -c $(SMATRIXL) $(SMATRIXLINC)

ifneq ($(ICC_MAJOR),)
# silence warning messages about subscripts being out of range
$(SMATRIXDO):   CXXFLAGS += -wd175 -I$(SMATRIXDIRI)
$(SMATRIXDO32): CXXFLAGS += -wd175 -I$(SMATRIXDIRI)
else
$(SMATRIXDO):   CXXFLAGS += -I$(SMATRIXDIRI)
$(SMATRIXDO32): CXXFLAGS += -I$(SMATRIXDIRI)
endif

all-$(MODNAME): $(SMATRIXLIB) $(SMATRIXMAP)

clean-$(MODNAME):
		@rm -f $(SMATRIXO) $(SMATRIXDO) $(SMATRIXDO32)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SMATRIXDEP) $(SMATRIXDS) $(SMATRIXDS32) $(SMATRIXDH) \
		   $(SMATRIXDH32) $(SMATRIXLIB) $(SMATRIXMAP)
		@rm -rf include/Math
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(SMATRIXDIRT)
else
		-@cd $(SMATRIXDIRT) && $(MAKE) distclean ROOTCONFIG=../../../bin/root-config
endif

distclean::     distclean-$(MODNAME)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(SMATRIXDIR)/test $(SMATRIXDIRT)
endif
		@cd $(SMATRIXDIRT) && $(MAKE) ROOTCONFIG=../../../bin/root-config
