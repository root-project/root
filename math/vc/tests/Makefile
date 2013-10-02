# Makefile for the ROOT test programs.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

#ROOTSYS = ../../../..
#include $(ROOTSYS)/etc/Makefile.arch
#include $(ROOTSYS)/config/Makefile.config


RC     := root-config
ifeq ($(shell which $(RC) 2>&1 | sed -ne "s@.*/$(RC)@$(RC)@p"),$(RC))
MKARCH := $(wildcard $(shell $(RC) --etcdir)/Makefile.arch)
RCONFIG := $(wildcard $(shell $(RC) --incdir)/RConfigure.h)
endif
ifneq ($(MKARCH),)
include $(MKARCH)
else
ifeq ($(ROOTSYS),)
ROOTSYS = ..
endif
include $(ROOTSYS)/etc/Makefile.arch
include $(ROOTSYS)/config/Makefile.config
endif

#------------------------------------------------------------------------------

# ifeq ($(PLATFORM),macosx) 
# #unroll loop better on gcc > 4
#CXXFLAGS+= -O3 -g  
# endif
#AVXCXXFLAG     := -mavx
#SIMDCXXFLAGS   := -mavx -msse4.2 -msse4.1 -msse4a -mssse3 -msse3 -msse2
#VCFLAGS        := -fabi-version=0  -Wno-unused-function

CXXFLAGS+=  $(VCFLAGS) 

ifeq ($(NOAVX),)
CXXFLAGS+= $(AVXCXXFLAG)
LDFLAGS += $(AVXCXXFLAG)
endif

EXTRALIBS += $(ROOTSYS)/lib/libVc.a


OBJS          = arithmetics.o casts.o deinterleave.o expandandmerge.o gather.o implicit_type_conversion.o   load.o \
		mask.o math.o memory.o scalaraccess.o scatter.o sse_blend.o stlcontainer.o store.o supportfunctions.o swizzles.o utils.o

PROGRAMS      = arithmetics  casts deinterleave expandandmerge gather implicit_type_conversion   load \
		mask math memory scalaraccess scatter sse_blend stlcontainer store supportfunctions swizzles utils


.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf)


arithmetics:   	arithmetics.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

casts: 		casts.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

deinterleave: 	deinterleave.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

expandandmerge:	expandandmerge.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

gather: 	gather.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

implicit_type_conversion: 	implicit_type_conversion.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

# implicit_type_conversion_failures: 	implicit_type_conversion_failures.o
# 		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
# 		@echo "$@ done"

# linkTest1: 	linkTest1.o
# 		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
# 		@echo "$@ done"

load:	 	load.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

mask:	 	mask.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

math:	 	math.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

memory:		memory.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

scalaraccess: 	scalaraccess.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

scatter:	scatter.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

sse_blend:	sse_blend.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

stlcontainer:	stlcontainer.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

store:		store.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

supportfunctions:	supportfunctions.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

swizzles:	swizzles.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"

utils:		utils.o
		$(LD) $(LDFLAGS) $^  $(EXTRALIBS)  $(OutPutOpt)$@
		@echo "$@ done"



all: 	$(PROGRAMS)


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
