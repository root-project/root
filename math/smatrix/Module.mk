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

SMATRIXDICTH:= Math/SMatrix.h \
		Math/SVector.h \
		Math/SMatrixDfwd.h \
		Math/SMatrixFfwd.h
#		Math/SMatrixD32fwd.h

SMATRIXDICTHINC := $(add-prefix include/,$(SMATRIXDICTH))

SMATRIXH1   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))
SMATRIXH2   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.icc))
SMATRIXH    := $(SMATRIXH1) $(SMATRIXH2)
SMATRIXS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SMATRIXO    := $(call stripsrc,$(SMATRIXS:.cxx=.o))

SMATRIXDEP  := $(SMATRIXO:.o=.d)  $(SMATRIXDO:.o=.d) $(SMATRIXDO32:.o=.d)

SMATRIXLIB  := $(LPATH)/libSmatrix.$(SOEXT)
SMATRIXMAP  := $(SMATRIXLIB:.$(SOEXT)=.rootmap)
SMATRIXMAP32:= $(SMATRIXLIB:.$(SOEXT)=32.rootmap)

# used in the main Makefile
SMATRIXH1_REL := $(patsubst $(MODDIRI)/Math/%,include/Math/%,$(SMATRIXH1))
SMATRIXH2_REL := $(patsubst $(MODDIRI)/Math/%,include/Math/%,$(SMATRIXH2))
ALLHDRS      += $(SMATRIXH1_REL) $(SMATRIXH2_REL)
ALLLIBS      += $(SMATRIXLIB)
ALLMAPS      += $(SMATRIXMAP) $(SMATRIXMAP32)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(SMATRIXH1_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(SMATRIXLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

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

$(call pcmrule,SMATRIX)
	$(noop)

$(SMATRIXDS):  $(SMATRIXDICTHINC) $(SMATRIXL) $(SMATRIXLINC) $(ROOTCLINGEXE) $(call pcmdep,SMATRIX)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SMATRIX) -c -writeEmptyRootPCM $(SMATRIXDICTH) $(SMATRIXL)

$(SMATRIXDS32): $(SMATRIXDICTHINC) $(SMATRIXL32) $(SMATRIXLINC) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ -multiDict $(subst -rmf $(SMATRIXMAP), -rmf $(SMATRIXMAP32),$(call dictModule,SMATRIX)) -c -writeEmptyRootPCM $(SMATRIXDICTH) $(SMATRIXL32)

$(SMATRIXMAP):  $(SMATRIXDICTHINC) $(SMATRIXL) $(SMATRIXLINC) $(ROOTCLINGEXE) $(call pcmdep,SMATRIX)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SMATRIXDS) $(call dictModule,SMATRIX) -c $(SMATRIXDICTH) $(SMATRIXL)

$(SMATRIXMAP32): $(SMATRIXDICTHINC) $(SMATRIXL32) $(SMATRIXLINC) $(ROOTCLINGEXE) $(call pcmdep,SMATRIX)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SMATRIXDS) $(subst -rmf $(SMATRIXMAP), -rmf $(SMATRIXMAP32),$(call dictModule,SMATRIX)) -c $(SMATRIXDICTH) $(SMATRIXL32)

ifneq ($(ICC_MAJOR),)
# silence warning messages about subscripts being out of range
$(SMATRIXDO):   CXXFLAGS += -wd175 -I$(SMATRIXDIRI)
$(SMATRIXDO32): CXXFLAGS += -wd175 -I$(SMATRIXDIRI)
else
$(SMATRIXDO):   CXXFLAGS += -I$(SMATRIXDIRI)
$(SMATRIXDO32): CXXFLAGS += -I$(SMATRIXDIRI)
endif

all-$(MODNAME): $(SMATRIXLIB)

clean-$(MODNAME):
		@rm -f $(SMATRIXO) $(SMATRIXDO) $(SMATRIXDO32)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SMATRIXDEP) $(SMATRIXDS) $(SMATRIXDS32) $(SMATRIXDH) \
		   $(SMATRIXDH32) $(SMATRIXLIB) $(SMATRIXMAP) $(SMATRIXMAP32)
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
