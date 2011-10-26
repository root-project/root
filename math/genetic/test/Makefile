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
LIBS        += "$(ROOTSYS)/lib/libGenetic.lib" "$(ROOTSYS)/lib/libTMVA.lib" \
               "$(ROOTSYS)/lib/libMLP.lib" "$(ROOTSYS)/lib/libTreePlayer.lib" \
               "$(ROOTSYS)/lib/libMinuit.lib" "$(ROOTSYS)/lib/libMinuit2.lib"
else
LIBS        += -lGenetic -lTMVA -lMLP -lTreePlayer -lMinuit -lMinuit2 -lThread -lXMLIO
CXXFLAGS += -g
endif

#CXXFLAGS += -D_GLIBCXX_PARALLEL -fopenmp
#LDFLAGS += -fopenmp 


TESTGAMINOBJ     = testGAMinimizer.$(ObjSuf)
TESTGAMINSRC     = testGAMinimizer.$(SrcSuf)
TESTGAMIN        = testGAMinimizer$(ExeSuf)

TIMEGAOBJ     = timeGA.$(ObjSuf)
TIMEGASRC     = timeGA.$(SrcSuf)
TIMEGA        = timeGA$(ExeSuf)

GATUTORIALOBJ     = GAMinTutorial.$(ObjSuf)
GATUTORIALSRC     = GAMinTutorial.$(SrcSuf)
GATUTORIAL        = GAMinTutorial$(ExeSuf)

OBJS          = $(TESTGAMINOBJ) $(TIMEGAOBJ)  $(GATUTORIALOBJ)

PROGRAMS      = $(TESTGAMIN) $(TIMEGA)  $(GATUTORIAL)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)

all:            $(PROGRAMS) 

$(TESTGAMIN):      $(TESTGAMINOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(TIMEGA):      $(TIMEGAOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

$(GATUTORIAL):      $(GATUTORIALOBJ)
		    $(LD) $(LDFLAGS) $^ $(LIBS)  $(OutPutOpt)$@
		    @echo "$@ done"

clean:
		@rm -f $(OBJS) core 

distclean:      clean
		@rm -f $(PROGRAMS) *.so


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
