SHELL = /bin/sh
NAME = all

ObjSuf = o
SrcSuf = cxx
DllSuf = so
OutPutOpt = -o

# ROOT includes
ROOTCFLAGS =  -I$(ROOTSYS)/include

INCLUDES = -I. $(ROOTCFLAGS)

include $(ROOTSYS)/test/Makefile.arch

FORT     := g77
FORTOPTS := $(FFLAGS) -g -c -O -DLINUX $(F77INCS) -fno-second-underscore
RANLIB   := ranlib
INCLUDES := $(INCLUDES) $(LINUX_SYS_INCLUDES)
SOCMD     = $(LD)

CXXSOURCES  = $(wildcard *.cxx)
CINTLIST   := $(addsuffix .h, $(basename $(CXXSOURCES)))
CINTLISTO  := $(addsuffix .o, $(basename $(CINTLIST)))
CPPSOURCES := $(CXXSOURCES) DataCint.cc
CXXOBJS     = $(addsuffix .o, $(basename $(CPPSOURCES)))
CXXSO      := libData.so


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
#.cc.o:
#	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<
#
#.cxx.o:
#	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

all: depend rootcint lib

rootcint: DataCint.cc DataCint.o

DataCint.cc: LinkDef.h $(CINTLIST)
	$(RM) DataCint.*
	@echo "Generating DataCint dictionary ..." $(CXXSORUCE)
	rootcint -f DataCint.cc -c -I. $(CINTLIST) LinkDef.h

DataCint.o: DataCint.cc
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

DataBlockBase.o : DataBlockBase.cxx DataBlockBase.h
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

PureAbstractInterface.o : PureAbstractInterface.cxx PureAbstractInterface.h
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

DataBlock1.o : DataBlock1.cxx DataBlock1.h PureAbstractInterface.h
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

DataBlock2.o : DataBlock2.cxx DataBlock2.h PureAbstractInterface.h
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<


lib: FORCE $(CXXSO)

$(CXXSO): $(CXXOBJS)
	$(SOCMD) $(SOFLAGS) $(CXXOBJS) $(OutPutOpt) $(CXXSO)

depend: FORCE
	@makedepend -f Makefile $(INCLUDES) $(CXXSOURCE)

tar: FORCE
	tar cvf badio.tar Makefile LinkDef.h $(CINTLIST) $(CXXSOURCES) *.C *.root
	gzip badio.tar

CLEAN_TARGETS += $(ALL_LIBRARIES) *.o *~ core a.out $(CXXSO) DataCint.* Makefile.bak *.tar*

TEST_TARGETS += read_test

include ../../Rules.mk

read_test.log:
	$(CMDECHO) root.exe -b -l -q read_test.C > read_test.log

read_test: read_test.log
	$(TestDiff)

FORCE:

# DO NOT DELETE
