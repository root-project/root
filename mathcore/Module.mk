# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODDIR       := mathcore
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHCOREDIR  := $(MODDIR)
MATHCOREDIRS := $(MATHCOREDIR)/src
MATHCOREDIRI := $(MATHCOREDIR)/inc

##### libMathCore #####
MATHCOREL    := $(MODDIRI)/Math/LinkDef.h 
MATHCORELINC := $(MODDIRI)/Math/LinkDef_Func.h \
                $(MODDIRI)/Math/LinkDef_GenVector.h \
                $(MODDIRI)/Math/LinkDef_Point3D.h \
                $(MODDIRI)/Math/LinkDef_Vector3D.h \
                $(MODDIRI)/Math/LinkDef_Vector4D.h \
                $(MODDIRI)/Math/LinkDef_Rotation.h 
MATHCOREDS   := $(MODDIRS)/G__MathCore.cxx
MATHCOREDO   := $(MATHCOREDS:.cxx=.o)
MATHCOREDH   := $(MATHCOREDS:.cxx=.h)

MATHCOREDH1  :=  $(MODDIRI)/Math/Vector3D.h \
                 $(MODDIRI)/Math/Point3D.h \
                 $(MODDIRI)/Math/Vector4D.h \
                 $(MODDIRI)/Math/Rotation3D.h \
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
                 $(MODDIRI)/Math/Plane3D.h \
                 $(MODDIRI)/Math/SpecFuncMathCore.h \
                 $(MODDIRI)/Math/ProbFuncMathCore.h \
                 $(MODDIRI)/Math/DistFunc.h \
                 $(MODDIRI)/Math/VectorUtil_Cint.h



MATHCOREAH   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))
MATHCOREGVH  := $(filter-out $(MODDIRI)/Math/GenVector/LinkDef%, $(wildcard $(MODDIRI)/Math/GenVector/*.h))
MATHCOREH    := $(MATHCOREAH) $(MATHCOREGVH)
MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO    := $(MATHCORES:.cxx=.o)

MATHCOREDEP  := $(MATHCOREO:.o=.d)  $(MATHCOREDO:.o=.d)

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(MATHCOREH))
ALLLIBS      += $(MATHCORELIB)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)

##### local rules #####
include/Math/%.h: $(MATHCOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math/GenVector" ]; then   \
		   mkdir -p include/Math/GenVector;       \
		fi)
		cp $< $@

$(MATHCORELIB): $(MATHCOREO) $(MATHCOREDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathCore.$(SOEXT) $@     \
		   "$(MATHCOREO) $(MATHCOREDO)"             \
		   "$(MATHCORELIBEXTRA)"

$(MATHCOREDS):  $(MATHCOREDH1) $(MATHCOREL) $(MATHCORELINC) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@echo "for files $(MATHCOREDH1)"
		$(ROOTCINTTMP) -f $@ -c $(MATHCOREDH1) $(MATHCOREL)

$(MATHCOREDO):  $(MATHCOREDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-mathcore:   $(MATHCORELIB)

map-mathcore:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(MATHCORELIB) \
		   -d $(MATHCORELIBDEP) -c $(MATHCOREL) $(MATHCORELINC)

map::           map-mathcore

clean-mathcore:
		@rm -f $(MATHCOREO) $(MATHCOREDO)

clean::         clean-mathcore

distclean-mathcore: clean-mathcore
		@rm -f $(MATHCOREDEP) $(MATHCOREDS) $(MATHCOREDH) $(MATHCORELIB)
		@rm -rf include/Math

distclean::     distclean-mathcore

test-mathcore:	all-mathcore
		@cd $(MATHCOREDIR)/test; make


