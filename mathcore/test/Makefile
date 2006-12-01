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
EXTRALIBS        = -include:_G__cpp_setupG__MathCore -include:_G__cpp_setupG__MathCore32 ../../lib/libMathCore.lib
else
EXTRALIBS        = -lMathCore
endif


CXXFLAGS += -g
# use reflex
#useReflex=yes
ifneq ($(useReflex),)
GCCXML=yes
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


all:            $(PROGRAMS) libTrackDict.$(DllSuf)


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

ifneq ($(GCCXML),)
TrackDict.$(SrcSuf): 	Track.h TrackDict.xml
			@echo "Generating dictionary $@ using gccxml ..."
			genreflex Track.h --selection_file=TrackDict.xml -o TrackDict.cxx -I$(ROOTSYS)/include
else
TrackDict.$(SrcSuf): 	Track.h TrackLinkDef.h
			@echo "Generating dictionary $@ using rootcint ..."
			rootcint -f $@ -c $^	
endif


libTrackDict.$(DllSuf): 	TrackDict.$(ObjSuf) 
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(GLIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(GLIBS) -p 0 $^
else
ifeq ($(PLATFORM),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ $(OutPutOpt) $@
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
	 	   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
else
ifeq ($(PLATFORM),win32)
		bindexplib libTrackDict $^ > libTrackDict.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)libTrackDict.lib
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ libTrackDict.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $^  $(OutPutOpt)$@
endif
endif
endif
endif




clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
