# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

#ROOTSYS = ../../../..
include $(ROOTSYS)/etc/Makefile.arch
include $(ROOTSYS)/config/Makefile.config

#------------------------------------------------------------------------------

# ifeq ($(PLATFORM),macosx) 
# #unroll loop better on gcc > 4
#CXXFLAGS+= -O3 -g  
# endif
AVXCXXFLAG     := -mavx
SIMDCXXFLAGS   := -mavx -msse4.2 -msse4.1 -msse4a -mssse3 -msse3 -msse2
VCFLAGS        := -fabi-version=0  -Wno-unused-function

CXXFLAGS+=  $(VCFLAGS) 


ifeq ($(NOAVX),)
CXXFLAGS+= $(AVXCXXFLAG)
LDFLAGS += $(AVXCXXFLAG)
endif
#CXXFLAGS+= -O3
#CXXFLAGS+= --fast-math
#CXXFLAGS += -ftree-vectorize
ifneq ($(AUTOVEC),)
CXXFLAGS+= -ftree-vectorize
endif
ifneq ($(OPT3),)
CXXFLAGS+= -O3
LDFLAGS += -O3
endif
ifneq ($(OPT2EXT),)
CXXFLAGS+= -finline-functions -ftree-vectorize -funswitch-loops -fgcse-after-reload -fipa-cp-clone -fpredictive-commoning -ftree-loop-distribute-patterns
LDFLAGS+= -finline-functions -ftree-vectorize -funswitch-loops -fgcse-after-reload -fipa-cp-clone -fpredictive-commoning  -ftree-loop-distribute-patterns
endif
ifneq ($(FASTM),)
CXXFLAGS+= -O3 --fast-math -ftree-vectorize -Ofast
LDFLAGS+=  -O3 --fast-math -ftree-vectorize -Ofast
endif
#
ifneq ($(USEVC),)
CXXFLAGS+= -DUSE_VC 
EXTRALIBS += $(ROOTSYS)/lib/libVc.a
ifneq ($(VCSCALAR),)
CXXFLAGS+= -DVC_IMPL=Scalar
endif
endif

ifneq ($(USEVDT),)
CXXFLAGS+= -DUSE_VDT -I/home/data/moneta/vdt_trunk/include
ifeq ($(NOAVX),)
EXTRALIBS += -L/home/data/moneta/vdt_trunk/lib-avx -lvdt 
else
EXTRALIBS += -L/home/data/moneta/vdt_trunk/lib-sse -lvdt 
endif
endif

ifneq ($(DEBUG),)
CXXFLAGS += -g -DDEBUG
endif

#for debugging vectorization
#CXXFLAGS+= -ftree-vectorizer-verbose=2


ifneq ($(NDIM1),)
CXXFLAGS += -DNDIM1=$(NDIM1) 
endif
ifneq ($(NDIM2),)
CXXFLAGS += -DNDIM2=$(NDIM2) 
endif


LIBS =  -L$(ROOTSYS)/lib -lCore  -lMathCore -lMatrix -lHist -lTree


TESTFITOBJ     = testFitPerf.$(ObjSuf)
TESTFITSRC     = testFitPerf.$(SrcSuf)
TESTFIT        = testFitPerf$(ExeSuf)


OBJS          = $(TESTFITOBJ)

PROGRAMS      = $(TESTFIT)


.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


$(TESTFIT):    $(TESTFITOBJ)
		$(LD) $(LDFLAGS) $^ $(LIBS) $(EXTRALIBS) $(OutPutOpt)$@
		@echo "$@ done"




check: 	all
	for prog in $(PROGRAMS); do \
	   ./$$prog > $$prog.out; \
	done; 

clean:
		@rm -f $(OBJS) $(PROGRAMS) 

distclean:      clean
		@rm -f $(PROGRAMS) 


.SUFFIXES: .$(SrcSuf)


.$(SrcSuf).$(ObjSuf):  
	$(CXX)  $(CXXFLAGS) -c $<
