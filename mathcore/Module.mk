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
MATHCOREL    := $(MODDIRI)/MathCore/LinkDef.h 
MATHCORELINC := $(MODDIRI)/MathCore/LinkDef_Func.h \
                $(MODDIRI)/MathCore/LinkDef_GenVector.h \
                $(MODDIRI)/MathCore/LinkDef_Point3D.h \
                $(MODDIRI)/MathCore/LinkDef_Vector3D.h \
                $(MODDIRI)/MathCore/LinkDef_Vector4D.h 
MATHCOREDS   := $(MODDIRS)/G__MathCore.cxx
MATHCOREDO   := $(MATHCOREDS:.cxx=.o)
MATHCOREDH   := $(MATHCOREDS:.cxx=.h)

MATHCOREDH1  := $(MODDIRI)/MathCore/Vector3D.h \
                $(MODDIRI)/MathCore/Point3D.h \
                $(MODDIRI)/MathCore/Vector4D.h \
                $(MODDIRI)/MathCore/Rotation3D.h \
                $(MODDIRI)/MathCore/SpecFunc.h \
                $(MODDIRI)/MathCore/DistFunc.h \
                $(MODDIRI)/MathCore/ProbFunc.h \
                $(MODDIRI)/MathCore/VectorUtil_Cint.h

MATHCOREH    := $(filter-out $(MODDIRI)/MathCore/LinkDef%, $(wildcard $(MODDIRI)/MathCore/*.h))
MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO    := $(MATHCORES:.cxx=.o)

MATHCOREDEP  := $(MATHCOREO:.o=.d)  $(MATHCOREDO:.o=.d)

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/MathCore/%.h,include/MathCore/%.h,$(MATHCOREH))
ALLLIBS      += $(MATHCORELIB)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)

##### local rules #####
include/MathCore/%.h: $(MATHCOREDIRI)/MathCore/%.h
		@(if [ ! -d "include/MathCore" ]; then   \
		   mkdir -p include/MathCore;            \
		fi)
		cp $< $@

$(MATHCORELIB): $(MATHCOREO) $(MATHCOREDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathCore.$(SOEXT) $@     \
		   "$(MATHCOREO) $(MATHCOREDO)"             \
		   "$(MATHCORELIBEXTRA)"

$(MATHCOREDS):  $(MATHCOREDH1) $(MATHCOREL) $(MATHCORELINC) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
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
		@rm -rf include/MathCore

distclean::     distclean-mathcore


