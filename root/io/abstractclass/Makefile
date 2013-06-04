SHELL = /bin/sh
NAME = all


# ROOT includes
ROOTCFLAGS =  -I$(ROOTSYS)/include

INCLUDES = -I. $(ROOTCFLAGS)

CLEAN_TARGETS += $(ALL_LIBRARIES) *.$(ObjSuf) *~ core a.out $(CXXSO) DataCint.* Makefile.bak *.tar* *.log

TEST_TARGETS += depend rootcint lib read_test abstractInVector

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

#include $(ROOTSYS)/test/Makefile.arch

FORT     := g77
FORTOPTS := $(FFLAGS) -g -c -O -DLINUX $(F77INCS) -fno-second-underscore
RANLIB   := ranlib
INCLUDES := $(INCLUDES) $(LINUX_SYS_INCLUDES)
SOCMD     = $(LD)

CXXSOURCES  = DataBlock1.cxx  DataBlock2.cxx  DataBlockBase.cxx  PureAbstractInterface.cxx
CINTLIST   := $(addsuffix .h, $(basename $(CXXSOURCES)))
CINTLISTO  := $(addsuffix .$(ObjSuf), $(basename $(CINTLIST)))
CPPSOURCES := $(CXXSOURCES) DataCint.cc
CXXOBJS     = $(addsuffix .$(ObjSuf), $(basename $(CPPSOURCES)))
CXXSO      := libData.$(DllSuf)


# LIBS should contain general libraries needed to compile programs. 
LIBS := $(GLIBS) $(SYSLIBS)
STATLIBS := $(LABYRINTHLIBTRIG)

# Default compiler and preprocessor flags
CXXFLAGS      := $(ENV_CXXFLAGS) $(CXXFLAGS)
CPPFLAGS       = $(INCLUDES)
CFLAGS         = -g
LDFLAGS       := $(LDFLAGS) $(LIBS)

# Define some rules.
#.SUFFIXES : .inc .cxx .cc
#
#.cc.$(ObjSuf):
#	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<
#
#.cxx.$(ObjSuf):
#	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

all: depend rootcint lib

rootcint: DataCint.cc DataCint.$(ObjSuf)

DataCint.cc: LinkDef.h $(CINTLIST)
	$(CMDECHO)$(RM) DataCint.*
#	$(CMDECHO)echo "Generating DataCint dictionary ..." $(CXXSORUCE)
	$(CMDECHO)rootcint -f DataCint.cc -c -I. $(CINTLIST) LinkDef.h

DataCint.$(ObjSuf): DataCint.cc

DataBlockBase.$(ObjSuf) : DataBlockBase.cxx DataBlockBase.h

PureAbstractInterface.$(ObjSuf) : PureAbstractInterface.cxx PureAbstractInterface.h

DataBlock1.$(ObjSuf) : DataBlock1.cxx DataBlock1.h PureAbstractInterface.h

DataBlock2.$(ObjSuf) : DataBlock2.cxx DataBlock2.h PureAbstractInterface.h


lib: FORCE $(CXXSO)

$(CXXSO): $(CXXOBJS)
	$(BuildFromObjs)
#	$(CMDECHO)$(SOCMD) $(SOFLAGS) $(CXXOBJS) $(OutPutOpt)$(CXXSO) $(EXPLLINKLIBS)

depend: FORCE
	$(CMDECHO)rmkdepend -f Makefile $(INCLUDES) $(CXXSOURCE)

tar: FORCE
	tar cvf badio.tar Makefile LinkDef.h $(CINTLIST) $(CXXSOURCES) *.C *.root
	gzip badio.tar

abstractInVector: abstractInVector_C.$(DllSuf)
	$(CMDECHO) test -e abstractInVector_C.$(DllSuf)  > /dev/null

read_test.log: $(CXXSO)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q read_test.C > read_test.log

read_test: read_test.log
	$(TestDiff)

FORCE:

# DO NOT DELETE
