# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ROOTSYS = ../../../..
include $(ROOTSYS)/etc/Makefile.arch

#------------------------------------------------------------------------------

ifeq ($(PLATFORM),win32)
LIBS        += "$(ROOTSYS)/lib/libMinuit2.lib"
else
LIBS        += -lMinuit2 
CXXFLAGS += -g      
endif

# for using with MPI
ifneq ($(USE_OPENMPI),)
CXX=mpic++
LD=mpic++
endif

QUAD1SRC    = Quad1FMain.$(SrcSuf)
QUAD1OBJ    = Quad1FMain.$(ObjSuf) 
QUAD1       = test_Minuit2_Quad1F$(ExeSuf)

QUAD4SRC    = Quad4FMain.$(SrcSuf)
QUAD4OBJ    = Quad4FMain.$(ObjSuf) 
QUAD4       = test_Minuit2_Quad4F$(ExeSuf)

QUAD8SRC    = Quad8FMain.$(SrcSuf)
QUAD8OBJ    = Quad8FMain.$(ObjSuf) 
QUAD8       = test_Minuit2_Quad8F$(ExeSuf)

QUAD12SRC    = Quad12FMain.$(SrcSuf)
QUAD12OBJ    = Quad12FMain.$(ObjSuf) 
QUAD12       = test_Minuit2_Quad12F$(ExeSuf)


OBJS          = $(QUAD1OBJ) $(QUAD4OBJ) $(QUAD8OBJ) $(QUAD12) 

PROGRAMS      = $(QUAD1) $(QUAD4) $(QUAD8) $(QUAD12)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)


$(QUAD1OBJ):    Quad1F.h
$(QUAD1): 	$(QUAD1OBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(QUAD4OBJ):  	Quad4F.h
$(QUAD4): 	$(QUAD4OBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(QUAD8OBJ):  	Quad8F.h
$(QUAD8): 	$(QUAD8OBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(QUAD12OBJ):  	Quad12F.h
$(QUAD12): 	$(QUAD12OBJ) 
		$(LD) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS)  *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
