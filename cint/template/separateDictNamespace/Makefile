
TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  *.o *.so *Dict* run.output

include ../../../root/Rules.mk

ROOTCFLAGS   := $(shell root-config --cflags) -DUSE_ROOT
ROOTLIBS     := $(patsubst -lCore, -lThread -lCore, $(shell root-config --libs))
ROOTLIBS     += -lGui
ROOTGLIBS    := $(shell root-config --glibs)

sepDictLibs: libmaster.so libslave1.so libslave2.so
	rm -f run.output
	@root.exe -q -l -b run.C 2>&1 | grep -v tagnum >| run.output && \
	diff run.good.output run.output

all: sepDictLibs

libmaster.so: Master.o MasterDict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave1.so: Slave1.o  Slave1Dict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave3a.so: Slave3a.o  Slave3aDict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave2.so: Slave2.o  Slave2Dict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@

libslave3.so: Slave3.o  Slave3Dict.o
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ -o $@


MasterDict.C: Master.hh MasterLinkDef.hh
Slave1Dict.C: Slave1.hh Master.hh Slave1LinkDef.hh
Slave2Dict.C: Master.hh Slave2.hh Slave2LinkDef.hh

%Dict.C:
	rootcint -f $@ -c $^ 

%Dict.o: %Dict.C	
	$(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@

%.o: %.C	
	$(CXX) $(CXXFLAGS) -I. $(ROOTCFLAGS) -c $< -o $@

clean_local:
	rm -f *.o *.so *Dict*