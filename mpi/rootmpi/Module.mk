# Module.mk for proofd module
# Copyright (c) 2012 Omar Andres Zapata Mesa
#
# Author: Omar Andres Zapata Mesa, 7/10/2012

MODNAME      := rootmpi
MODDIR       := $(ROOT_SRCDIR)/mpi/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTMPIDIR    := $(MODDIR)
ROOTMPIDIRS   := $(ROOTMPIDIR)/src
ROOTMPIDIRI   := $(ROOTMPIDIR)/inc


##### rootmpi #####
ROOTMPIEXEH   := $(MODDIRI)/TRun.h
ROOTMPIEXES   := $(MODDIRS)/rootmpi.cpp $(MODDIRS)/TRun.cpp 
ROOTMPIEXEO   := $(call stripsrc,$(ROOTMPIEXES:.cpp=.o))
ROOTMPIDEP    := $(ROOTMPIEXEO:.o=.d)
ROOTMPIEXE    := bin/rootmpi

ROOTMPIEXELIBS := -Llib -lCore -lRint -lCint -lRIO -lThread -lNet -lMathCore

ROOTMPIEXEDEPS := $(CORELIB) $(RINTLIB) $(CINTLIB) $(THREAD) $(NETLIB) $(IOLIB) $(MATHCORELIB)

$(ROOTMPIEXE):   $(ROOTMPIEXEO) $(ROOTMPIEXEDEPS)
		$(CXX) $(ROOTMPIEXEO) -o $@  $(LDFLAGS) $(ROOTMPIEXELIBS)




all-$(MODNAME): $(ROOTMPIEXE)

clean-$(MODNAME):
		@rm -f $(ROOTMPIEXEO) 

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTMPIEXEO) $(ROOTMPIEXE)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(ROOTMPIEXEO): CXXFLAGS += -I$(ROOTMPIDIRI) -Iinclude 

ALLEXECS     += $(ROOTMPIEXE)
