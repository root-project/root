
#CC  := gcc
#CXX := g++
#F77 := g77
#LD  := g++
#SOFLAGS   = -Wl,-soname,${@F} -shared

TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  *.o *.so *Dict* run.output

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ROOTCFLAGS   := $(shell root-config --cflags) -DUSE_ROOT
ROOTLIBS     := $(patsubst -lCore, -lThread -lCore, $(shell root-config --libs))
ROOTLIBS     += -lGui
ROOTGLIBS    := $(shell root-config --glibs)

sepDictLibs: run.output
	$(CMDECHO) diff run.good.output run.output


run.output: libmaster.so libslave1.so libslave2.so 
	$(CMDECHO)root.exe -q -l -b run.C 2>&1 | grep -v tagnum >| run.output

all: sepDictLibs


libmaster.so: MyClass.o MasterDict.o
	$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave1.so:  Slave1Dict.o
	$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave2.so:  Slave2Dict.o
	$(CMDECHO)$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

MasterDict.C: MyClass.hh MasterLinkDef.hh
Slave1Dict.C: MyClass.hh Slave1LinkDef.hh
Slave2Dict.C: MyClass.hh Slave2LinkDef.hh

%Dict.C:
	$(CMDECHO)rootcint -f $@ -c $^ 

%Dict.o: %Dict.C	
	$(CMDECHO)$(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@ >& $@_build.log

%.o: %.C	
	$(CMDECHO)$(CXX) $(CXXFLAGS) -I. $(ROOTCFLAGS) -c $< -o $@ >& $@_build.log

clean_sepdict:
	$(CMDECHO)rm -f *.o *.so *Dict*
