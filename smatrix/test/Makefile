# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

include $(ROOTSYS)/test/Makefile.arch
include $(ROOTSYS)/config/Makefile.config

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),macosx) 
#unroll loop better on gcc > 4
CXXFLAGS+= -O3 -g  -funroll-loops 
endif



# if have clhep
#CLHEPBASE=/Users/moneta/mathlibs/CLHEP-1.9.2.2
ifneq ($(CLHEPBASE),)
CXXFLAGS+=  -I$(CLHEPBASE)/include -DHAVE_CLHEP
EXTRALIBS+=  $(CLHEPBASE)/lib/libCLHEP.a
endif

ifneq ($(PLATFORM),win32) 
LIBM = -lm
endif

ifneq ($(useReflex),)
GCCXML=yes
CXXFLAGS+=-DUSE_REFLEX -g
ifeq ($(PLATFORM),win32)
EXTRAIOLIBS = "$(ROOTSYS)/lib/libReflex.lib"  "$(ROOTSYS)/lib/libCintex.lib"  
else
EXTRAIOLIBS = -lReflex -lCintex
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


STRESSOPERATIONSOBJ     = stressOperations.$(ObjSuf)
STRESSOPERATIONSSRC     = stressOperations.$(SrcSuf)
STRESSOPERATIONS        = stressOperations$(ExeSuf)


STRESSKALMANOBJ     = stressKalman.$(ObjSuf)
STRESSKALMANSRC     = stressKalman.$(SrcSuf)
STRESSKALMAN        = stressKalman$(ExeSuf)


OBJS          = $(TESTSMATRIXOBJ) $(TESTOPERATIONSOBJ) $(TESTKALMANOBJ) $(TESTIOOBJ) $(STRESSOPERATIONSOBJ) $(STRESSKALMANOBJ) 


PROGRAMS      = $(TESTSMATRIX)  $(TESTOPERATIONS) $(TESTKALMAN) $(TESTIO) $(STRESSOPERATIONS) $(STRESSKALMAN) 


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

$(TESTIO):        $(TESTIOOBJ) 
		    $(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(EXTRAIOLIBS) $(OutPutOpt)$@
		    @echo "$@ done"

ifneq ($(useReflex),)
TrackDict.$(SrcSuf): 	Track.h TrackDict.xml
			@echo "Generating dictionary $@ using gccxml ..."
			#rootcint -gccxml -f $@ -c $^	
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
		@rm -f $(OBJS) $(PROGRAMS) core 

distclean:      clean
		@rm -f $(PROGRAMS) kalman.root stressOperations.root 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):  
	$(CXX)  $(CXXFLAGS) -c $<
