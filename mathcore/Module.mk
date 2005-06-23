# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := mathcore
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHCOREDIR    := $(MODDIR)
MATHCOREDIRS   := $(MATHCOREDIR)/src
MATHCOREDIRI   := $(MATHCOREDIR)/inc

##### libMathcore #####
# for dict
MATHCOREL     := $(MODDIRI)/GenVector/LinkDef.h
MATHCOREDS    := $(MODDIRS)/G__MathCore.cxx
MATHCOREDO    := $(MATHCOREDS:.cxx=.o)
MATHCOREDH    := $(MATHCOREDS:.cxx=.h)

MATHCOREDICTH := $(MODDIRI)/GenVector/Vector3D.h $(MODDIRI)/GenVector/Point3D.h $(MODDIRI)/GenVector/LorentzVector.h $(MODDIRI)/GenVector/VectorUtil_Cint.h 


MATHCOREAH     := $(filter-out $(MODDIRI)/LinkDef%, $(wildcard $(MODDIRI)/GenVector/*.h))
#MATHCOREBH     := $(wildcard $(MODDIRI)/Mathcore/Builder/*.h)
MATHCOREH      := $(MATHCOREAH) $(MATHCOREBH)
MATHCORES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO      := $(MATHCORES:.cxx=.o)

MATHCOREDEP    := $(MATHCOREO:.o=.d)  $(MATHCOREDO:.o=.d)

MATHCORELIBNAME := libMathCore.$(SOEXT)
MATHCORELIB    := $(LPATH)/$(MATHCORELIBNAME)
 
# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/GenVector/%.h,include/GenVector/%.h,$(MATHCOREH))
ALLLIBS      += $(MATHCORELIB)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)

##### local rules #####
include/GenVector/%.h: $(MATHCOREDIRI)/GenVector/%.h 
		@ ( if [ ! -d "include/GenVector" ] ;         \
		    then mkdir include/GenVector; fi ) ;      \
		cp $< $@


$(MATHCORELIB):   $(MATHCOREO) $(MATHCOREDO) $(MAINLIBS)
		$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"     \
		"$(SOFLAGS)" $(MATHCORELIBNAME) $@ "$(MATHCOREO)" "$(MATHCOREDO)"\
		"$(MATHCORELIBDIR) $(MATHCORECLILIB)"


$(MATHCOREDS):   $(MATHCOREH) $(MATHCOREL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MATHCOREDICTH) $(MATHCOREL)

$(MATHCOREDO):   $(MATHCOREDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<



all-mathcore:    $(MATHCORELIB)
		echo $(MATHCOREO)

map-mathcore:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(MATHCORELIB) \
		-d $(MATHCORELIBDEP) -c $(MATHCOREL)

map::           map-mathcore

clean-mathcore:
		@rm -f $(MATHCOREO) $(MATHCOREDO) 

clean::         clean-mathcore

distclean-mathcore: clean-mathcore
		@rm -f $(MATHCOREDEP) $(MATHCOREDS) $(MATHCOREDH) $(MATHCORELIB)

distclean::     distclean-mathcore

##### extra rules ######
#$(MATHCOREO): %.o: %.cxx
#		$(CXX) $(OPT) $(CXXFLAGS) $(MATHCOREINCDIR:%=-I%) -o $@ -c $<

