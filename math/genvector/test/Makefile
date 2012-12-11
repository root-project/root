# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../..
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
EXTRALIBS        = -include:_G__cpp_setupG__GenVector -include:_G__cpp_setupG__GenVector32 "$(ROOTSYS)/lib/libGenVector.lib" 
else
EXTRALIBS        = -lGenVector 
CXXFLAGS += -g
endif

ifeq ($(DIM),2)
CXXFLAGS += -DDIM_2
endif
ifeq ($(DIM),3)
CXXFLAGS += -DDIM_3
endif
ifneq ($(USE_CLHEP),)
CXXFLAGS += -DUSE_CLHEP
endif
ifneq ($(USE_ROOT),)
CXXFLAGS += -DUSE_ROOT
endif


# use reflex
#useReflex=yes
ifneq ($(USE_REFLEX),)
GCCXML=yes
CXXFLAGS += -DUSE_REFLEX
ifeq ($(PLATFORM),win32)
EXTRAIOLIBS = "$(ROOTSYS)/lib/libReflex.lib"  "$(ROOTSYS)/lib/libCintex.lib"  
else
EXTRAIOLIBS = -lReflex -lCintex
endif
endif


# if have clhep
ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
ifeq ($(PLATFORM),win32)
EXTRALIBS += "$(CLHEPBASE)/lib/CLHEP.lib"
else
#EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
EXTRALIBS+=  -L$(CLHEPBASE)/lib -lCLHEP
endif
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

STRESS2DOBJ     = stress2D.$(ObjSuf)
STRESS2DSRC     = stress2D.$(SrcSuf)
STRESS2D        = stress2D$(ExeSuf)

VECTOROPOBJ     = vectorOperation.$(ObjSuf)
VECTOROPSRC     = vectorOperation.$(SrcSuf)
VECTOROP        = vectorOperation$(ExeSuf)

#VECTORSCALEOBJ     = testVectorScale.$(ObjSuf)
#VECTORSCALESRC     = testVectorScale.$(SrcSuf)
#VECTORSCALE        = testVectorScale$(ExeSuf)


OBJS          = $(COORDINATES3DOBJ) $(COORDINATES4DOBJ) $(ROTATIONOBJ) $(BOOSTOBJ) $(GENVECTOROBJ) $(VECTORIOOBJ) $(STRESS3DOBJ) $(STRESS2DOBJ) $(ITERATOROBJ) $(VECTOROPOBJ) 


PROGRAMS      = $(COORDINATES3D)  $(COORDINATES4D) $(ROTATION) $(BOOST) $(GENVECTOR) $(VECTORIO)  $(STRESS3D) $(STRESS2D) $(ITERATOR) $(VECTOROP) 


		  
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

$(VECTORIO):   	  $(VECTORIOOBJ) libTrackDict.$(DllSuf)
		    $(LD) $(LDFLAGS) $(VECTORIOOBJ) $(LIBS) $(EXTRALIBS) $(EXTRAIOLIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESS3D):      $(STRESS3DOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESS2D):      $(STRESS2DOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(ITERATOR):           $(ITERATOROBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"


$(VECTOROP):   	$(VECTOROPOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(EXTRAIOLIBS) $(OutPutOpt)$@
		    @echo "$@ done"

# $(VECTORSCALE):   	$(VECTORSCALEOBJ)
# 		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(EXTRAIOLIBS) $(OutPutOpt)$@
# 		    @echo "$@ done"


ifneq ($(USE_REFLEX),)
TrackDict.$(SrcSuf): 	Track.h TrackDict.xml
			@echo "Generating dictionary $@ using gccxml ..."
			genreflex Track.h --selection_file=TrackDict.xml -o TrackDict.cxx -I$(ROOTSYS)/include
else
TrackDict.$(SrcSuf): 	Track.h TrackLinkDef.h
			@echo "Generating dictionary $@ using rootcling ..."
			$(ROOTCLING) -f $@ -c $^	
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
ifneq ($(subst $(MACOSX_MINOR),,1234),1234)
ifeq ($(MACOSX_MINOR),4)
		ln -sf $@ $(subst .$(DllSuf),.so,$@)
else
		$(LD) -bundle -undefined $(UNDEFOPT) $(LDFLAGS) $^ \
	 	   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
endif
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
		@rm -f $(OBJS) core TrackDict.$(SrcSuf) TrackDict.h

distclean:      clean
		@rm -f $(PROGRAMS) libTrackDict.*


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
