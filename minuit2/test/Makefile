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
EXTRALIBS        = "$(ROOTSYS)/lib/libMinuit2.lib"
else
EXTRALIBS        = -lMinuit2 
CXXFLAGS += -g      
endif

USERFUNCOBJ      = testUserFunc.$(ObjSuf)
USERFUNCSRC      = testUserFunc.$(SrcSuf)
USERFUNC         = testUserFunc$(ExeSuf)


GRAPHOBJ       = testGraph.$(ObjSuf)
GRAPHSRC       = testGraph.$(SrcSuf)
GRAPH          = testGraph$(ExeSuf)

MINIMIZEOBJ      = testMinimize.$(ObjSuf)
MINIMIZESRC      = testMinimize.$(SrcSuf)
MINIMIZE         = testMinimize$(ExeSuf)


OBJS          = $(USERFUNCOBJ)  $(GRAPHOBJ) $(MINIMIZEOBJ)

PROGRAMS      = $(USERFUNC)  $(GRAPH) $(MINIMIZE)

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


all:            $(PROGRAMS)

$(USERFUNC):    $(USERFUNCOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(GRAPH): 	$(GRAPHOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"

$(MINIMIZE): 	$(MINIMIZEOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"



clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS)  *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d .def so_locations

.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
