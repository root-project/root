# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODNAME       := genvector
MODDIR        := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

GENVECTORDIR  := $(MODDIR)
GENVECTORDIRS := $(GENVECTORDIR)/src
GENVECTORDIRI := $(GENVECTORDIR)/inc
GENVECTORDIRT := $(call stripsrc,$(GENVECTORDIR)/test)

##### libGenvector #####
GENVECTORL    := $(MODDIRI)/Math/LinkDef_GenVector.h
GENVECTORL32  := $(MODDIRI)/Math/LinkDef_GenVector32.h
GENVECTORLINC :=  \
                $(MODDIRI)/Math/LinkDef_Point3D.h \
                $(MODDIRI)/Math/LinkDef_Vector3D.h \
                $(MODDIRI)/Math/LinkDef_Vector4D.h \
                $(MODDIRI)/Math/LinkDef_GenVector2.h \
                $(MODDIRI)/Math/LinkDef_Rotation.h
GENVECTORDS   := $(call stripsrc,$(MODDIRS)/G__GenVector.cxx)
GENVECTORDS32 := $(call stripsrc,$(MODDIRS)/G__GenVector32.cxx)
GENVECTORDO   := $(GENVECTORDS:.cxx=.o)
GENVECTORDO32 := $(GENVECTORDS32:.cxx=.o)
GENVECTORDH   := $(GENVECTORDS:.cxx=.h)
GENVECTORDH32 := $(GENVECTORDS32:.cxx=.h)

GENVECTORDH1  := $(MODDIRI)/Math/Vector2D.h \
                 $(MODDIRI)/Math/Point2D.h \
                 $(MODDIRI)/Math/Vector3D.h \
                 $(MODDIRI)/Math/Point3D.h \
                 $(MODDIRI)/Math/Vector4D.h \
                 $(MODDIRI)/Math/Rotation3D.h \
                 $(MODDIRI)/Math/RotationZYX.h \
                 $(MODDIRI)/Math/RotationX.h \
                 $(MODDIRI)/Math/RotationY.h \
                 $(MODDIRI)/Math/RotationZ.h \
                 $(MODDIRI)/Math/LorentzRotation.h \
                 $(MODDIRI)/Math/Boost.h    \
                 $(MODDIRI)/Math/BoostX.h    \
                 $(MODDIRI)/Math/BoostY.h    \
                 $(MODDIRI)/Math/BoostZ.h    \
                 $(MODDIRI)/Math/EulerAngles.h \
                 $(MODDIRI)/Math/AxisAngle.h \
                 $(MODDIRI)/Math/Quaternion.h \
                 $(MODDIRI)/Math/Transform3D.h \
                 $(MODDIRI)/Math/Translation3D.h \
                 $(MODDIRI)/Math/Plane3D.h \
                 $(MODDIRI)/Math/VectorUtil.h

GENVECTORDH132:= $(MODDIRI)/Math/Vector2D.h \
	         $(MODDIRI)/Math/Point2D.h \
	         $(MODDIRI)/Math/Vector3D.h \
                 $(MODDIRI)/Math/Point3D.h \
                 $(MODDIRI)/Math/Vector4D.h \

GENVECTORAH   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
GENVECTORGVH  := $(filter-out $(MODDIRI)/Math/GenVector/LinkDef%, $(wildcard $(MODDIRI)/Math/GenVector/*.h))
GENVECTORH    := $(GENVECTORAH) $(GENVECTORGVH)
GENVECTORS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GENVECTORO    := $(call stripsrc,$(GENVECTORS:.cxx=.o))

GENVECTORDEP  := $(GENVECTORO:.o=.d) $(GENVECTORDO:.o=.d) $(GENVECTORDO32:.o=.d)

GENVECTORLIB  := $(LPATH)/libGenVector.$(SOEXT)
GENVECTORMAP  := $(GENVECTORLIB:.$(SOEXT)=.rootmap)
GENVECTORMAP32:= $(GENVECTORLIB:.$(SOEXT)=32.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(GENVECTORH))
ALLLIBS      += $(GENVECTORLIB)
ALLMAPS      += $(GENVECTORMAP) $(GENVECTORMAP32)

# include all dependency files
INCLUDEFILES += $(GENVECTORDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/Math/%.h: $(GENVECTORDIRI)/Math/%.h
		@(if [ ! -d "include/Math/GenVector" ]; then   \
		   mkdir -p include/Math/GenVector;       \
		fi)
		cp $< $@

# build lib genvector: use also obj from math and fit directory
$(GENVECTORLIB): $(GENVECTORO) $(GENVECTORDO) $(GENVECTORDO32) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libGenVector.$(SOEXT) $@     \
		   "$(GENVECTORO) $(GENVECTORDO) $(GENVECTORDO32)"    \
		   "$(GENVECTORLIBEXTRA)"

$(call pcmrule,GENVECTOR)
	$(noop)

$(GENVECTORDS):  $(GENVECTORDH1) $(GENVECTORL) $(GENVECTORLINC) $(ROOTCLINGEXE) $(call pcmdep,GENVECTOR)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GENVECTOR) -c -writeEmptyRootPCM $(GENVECTORDH1) $(GENVECTORL)

$(GENVECTORDS32): $(GENVECTORDH132) $(GENVECTORL32) $(GENVECTORLINC) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ -multiDict $(subst -rmf $(GENVECTORMAP), -rmf $(GENVECTORMAP32),$(call dictModule,GENVECTOR)) -c -writeEmptyRootPCM $(GENVECTORDH132) $(GENVECTORL32)

$(GENVECTORMAP):  $(GENVECTORDH1) $(GENVECTORL) $(GENVECTORLINC) $(ROOTCLINGEXE) $(call pcmdep,GENVECTOR)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GENVECTORDS) $(call dictModule,GENVECTOR) -c $(GENVECTORDH1) $(GENVECTORL)

$(GENVECTORMAP32): $(GENVECTORDH1) $(GENVECTORL) $(GENVECTORLINC32) $(ROOTCLINGEXE) $(call pcmdep,GENVECTOR)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GENVECTORDS) $(subst -rmf $(GENVECTORMAP), -rmf $(GENVECTORMAP32),$(call dictModule,GENVECTOR)) -c $(GENVECTORDH1) $(GENVECTORL32)

all-$(MODNAME): $(GENVECTORLIB) $(GENVECTORMAP) $(GENVECTORMAP32)

clean-$(MODNAME):
		@rm -f $(GENVECTORO) $(GENVECTORDO) $(GENVECTORDO32)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GENVECTORDEP) $(GENVECTORDS) $(GENVECTORDS32) \
		   $(GENVECTORDH) $(GENVECTORDH32) \
		   $(GENVECTORLIB) $(GENVECTORMAP) $(GENVECTORMAP32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(GENVECTORDIRT)
else
		@cd $(GENVECTORDIRT) && $(MAKE) distclean ROOTCONFIG=../../../bin/root-config
endif

distclean::     distclean-$(MODNAME)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(GENVECTORDIR)/test $(GENVECTORDIRT)
endif
		@cd $(GENVECTORDIRT) && $(MAKE) ROOTCONFIG=../../../bin/root-config

##### extra rules ######

# Optimize dictionary with stl containers.
$(GENVECTORDO): NOOPT = $(OPT)
$(GENVECTORDO32): NOOPT = $(OPT)
