
#CC  := gcc
#CXX := g++
#F77 := g77
#LD  := g++
#SOFLAGS   = -Wl,-soname,${@F} -shared

include ../../../root/Rules.mk

TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  *.o *.so *Dict* run.output

ROOTCFLAGS   := $(shell root-config --cflags) -DUSE_ROOT
ROOTLIBS     := $(patsubst -lCore, -lThread -lCore, $(shell root-config --libs))
ROOTLIBS     += -lGui
ROOTGLIBS    := $(shell root-config --glibs)

sepDictLibs: libmaster.so libslave1.so libslave2.so
	root.exe -q -l -b run.C | grep -v tagnum >| run.output && \
	diff run.good.output run.output

all: sepDictLibs
test: sepDictLibs tests

libmaster.so: MyClass.o MasterDict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave1.so:  Slave1Dict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave2.so:  Slave2Dict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

MasterDict.C: MyClass.hh MasterLinkDef.hh
Slave1Dict.C: MyClass.hh Slave1LinkDef.hh
Slave2Dict.C: MyClass.hh Slave2LinkDef.hh

%Dict.C:
	rootcint -f $@ -c $^ 

%Dict.o: %Dict.C	
	$(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@

%.o: %.C	
	$(CXX) $(CXXFLAGS) -I. $(ROOTCFLAGS) -c $< -o $@

clean_sepdict:
	rm -f *.o *.so *Dict*