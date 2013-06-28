# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ifeq ($(ROOTSYS),)
ROOTSYS = ../../..
endif
include $(ROOTSYS)/etc/Makefile.arch
include $(ROOTSYS)/config/Makefile.config

#------------------------------------------------------------------------------

# ifeq ($(PLATFORM),macosx) 
# #unroll loop better on gcc > 4
# CXXFLAGS+= -O3 -g  -funroll-loops 
# endif



# if have clhep
#CLHEPBASE=/Users/moneta/mathlibs/CLHEP-1.9.2.2
ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
ifeq ($(PLATFORM),win32)
EXTRALIBS += "$(CLHEPBASE)/lib/CLHEP.lib"
else
#EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
EXTRALIBS+=  -L$(CLHEPBASE)/lib -lCLHEP
endif
endif

ifneq ($(PLATFORM),win32) 
LIBM = -lm
endif

ifneq ($(USE_REFLEX),)
GCCXML=yes
CXXFLAGS+=-DUSE_REFLEX
ifeq ($(PLATFORM),win32)
EXTRALIBS += "$(ROOTSYS)/lib/libReflex.lib"  "$(ROOTSYS)/lib/libCintex.lib"  
else
EXTRALIBS += -lReflex -lCintex
endif
endif



TESTSMATRIXOBJ     = testSMatrix.$(ObjSuf)
TESTSMATRIXSRC     = testSMatrix.$(SrcSuf)
TESTSMATRIX        = testSMatrix$(ExeSuf)

TESTOPERATIONSOBJ     = testOperations.$(ObjSuf)
TESTOPERATIONSSRC     = testOperations.$(SrcSuf)  
TESTOPERATIONS        = testOperations$(ExeSuf)

TESTKALMANOBJ     = testKalman.$(ObjSuf)
TESTKALMANSRC     = testKalman.$(SrcSuf)
TESTKALMAN        = testKalman$(ExeSuf)

TESTIOOBJ     = testIO.$(ObjSuf) 
TESTIOSRC     = testIO.$(SrcSuf) 
TESTIO        = testIO$(ExeSuf) 

TESTINVERSIONOBJ     = testInversion.$(ObjSuf)
TESTINVERSIONSRC     = testInversion.$(SrcSuf)  
TESTINVERSION        = testInversion$(ExeSuf)


STRESSOPERATIONSOBJ     = stressOperations.$(ObjSuf)
STRESSOPERATIONSSRC     = stressOperations.$(SrcSuf)
STRESSOPERATIONS        = stressOperations$(ExeSuf)


STRESSKALMANOBJ     = stressKalman.$(ObjSuf)
STRESSKALMANSRC     = stressKalman.$(SrcSuf)
STRESSKALMAN        = stressKalman$(ExeSuf)


OBJS          = $(TESTSMATRIXOBJ) $(TESTOPERATIONSOBJ) $(TESTKALMANOBJ) $(TESTINVERSIONOBJ) $(TESTIOOBJ)  $(STRESSOPERATIONSOBJ) $(STRESSKALMANOBJ) 


PROGRAMS      = $(TESTSMATRIX)  $(TESTOPERATIONS) $(TESTKALMAN) $(TESTINVERSION) $(TESTIO) $(STRESSOPERATIONS) $(STRESSKALMAN) 


.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS) libTrackDict.$(DllSuf)

testOperations.$(ObjSuf): matrix_op.h matrix_util.h TestTimer.h

testKalman.$(ObjSuf): matrix_util.h TestTimer.h

stressOperations.$(ObjSuf): $(TESTOPERATIONSOBJ)


$(TESTSMATRIX):   $(TESTSMATRIXOBJ)
		    $(LD) $(LDFLAGS) $^  $(LIBM) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTOPERATIONS):   $(TESTOPERATIONSOBJ) 
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTKALMAN):    $(TESTKALMANOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTINVERSION): $(TESTINVERSIONOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(TESTIO):        $(TESTIOOBJ) libTrackDict.$(DllSuf)
		    $(LD) $(LDFLAGS) $(TESTIOOBJ) $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

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
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
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
		$(LD) $(SOFLAGS) $(LDFLAGS) $^  $(LIBS) $(OutPutOpt)$@
		$(MT_DLL) 
endif
endif
endif


$(STRESSKALMAN):  $(STRESSKALMANOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

$(STRESSOPERATIONS):  $(STRESSOPERATIONSOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		    @echo "$@ done"

check: 	all
	for prog in $(PROGRAMS); do \
	   ./$$prog > $$prog.out; \
	done; 

clean:
		@rm -f $(OBJS) $(PROGRAMS) libTrackDict.* TrackDict.$(ObjSuf) TrackDict.$(SrcSuf) TrackDict.h core 

distclean:      clean
		@rm -f $(PROGRAMS) kalman.root stressOperations.root 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):  
	$(CXX)  $(CXXFLAGS) -c $<
