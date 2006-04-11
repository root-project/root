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
EXTRALIBS        = $(ROOTSYS)/lib/libMathCore.lib
else
EXTRALIBS        = -lMathCore
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

GENVECTOROBJ     = testGenVector.$(ObjSuf)
GENVECTORSRC     = testGenVector.$(SrcSuf)
GENVECTOR        = testGenVector$(ExeSuf)


OBJS          = $(COORDINATES3DOBJ) $(COORDINATES4DOBJ) $(ROTATIONOBJ) $(BOOSTOBJ) $(GENVECTOROBJ)


PROGRAMS      = $(COORDINATES3D)  $(COORDINATES4D) $(ROTATION) $(BOOST) $(GENVECTOR)

		  
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



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
