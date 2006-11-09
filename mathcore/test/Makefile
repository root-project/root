# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
EXTRALIBS        = "$(ROOTSYS)/lib/libMathCore.lib"
else
EXTRALIBS        = -lMathCore
endif

# use reflex
#useReflex=yes
ifneq ($(useReflex),)
CXXFLAGS += -DUSE_REFLEX
EXTRAIOLIBS = -lReflex -lCintex
endif


COORDINATES3DOBJ     = coordinates3D.$(ObjSuf)
COORDINATES3DSRC     = coordinates3D.$(SrcSuf)
COORDINATES3D        = coordinates3D$(ExeSuf)

COORDINATES4DOBJ     = coordinates4D.$(ObjSuf)
COORDINATES4DSRC     = coordinates4D.$(SrcSuf)
COORDINATES4D        = coordinates4D$(ExeSuf)

ROTATIONOBJ     = rotationApplication.$(ObjSuf)
ROTATIONSRC     = rotationApplication.$(SrcSuf)
ROTATION        = rotationApplication$(ExeSuf)

BOOSTOBJ     = testBoost.$(ObjSuf)
BOOSTSRC     = testBoost.$(SrcSuf)
BOOST        = testBoost$(ExeSuf)

ITERATOROBJ     = testIterator.$(ObjSuf)
ITERATORSRC     = testIterator.$(SrcSuf)
ITERATOR        = testIterator$(ExeSuf)

GENVECTOROBJ     = testGenVector.$(ObjSuf)
GENVECTORSRC     = testGenVector.$(SrcSuf)
GENVECTOR        = testGenVector$(ExeSuf)

VECTORIOOBJ     = testVectorIO.$(ObjSuf)
VECTORIOSRC     = testVectorIO.$(SrcSuf)
VECTORIO        = testVectorIO$(ExeSuf)

STRESS3DOBJ     = stress3D.$(ObjSuf)
STRESS3DSRC     = stress3D.$(SrcSuf)
STRESS3D        = stress3D$(ExeSuf)



OBJS          = $(COORDINATES3DOBJ) $(COORDINATES4DOBJ) $(ROTATIONOBJ) $(BOOSTOBJ) $(GENVECTOROBJ) $(VECTORIOOBJ) $(STRESS3DOBJ) $(ITERATOROBJ)


PROGRAMS      = $(COORDINATES3D)  $(COORDINATES4D) $(ROTATION) $(BOOST) $(GENVECTOR) $(VECTORIO) $(STRESS3D) $(ITERATOR)


		  
.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)


$(COORDINATES3D):   $(COORDINATES3DOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(COORDINATES4D):   $(COORDINATES4DOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(ROTATION):        $(ROTATIONOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(BOOST):           $(BOOSTOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(GENVECTOR):     $(GENVECTOROBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(VECTORIO):   	  $(VECTORIOOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(EXTRAIOLIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESS3D):      $(STRESS3DOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(ITERATOR):           $(ITERATOROBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"


clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
