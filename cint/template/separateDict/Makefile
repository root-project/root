
ARCH          = $(shell root-config --arch)
PLATFORM      = $(ARCH)

TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  *.$(ObjSuf) *.$(DllSuf) *Dict* *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ROOTCFLAGS   := $(shell root-config --cflags) -DUSE_ROOT
ROOTLIBS     := $(patsubst -lCore, -lThread -lCore, $(shell root-config --libs))
ROOTLIBS     += -lGui
ROOTGLIBS    := $(shell root-config --glibs)

sepDictLibs: sepDictLibs.log
	$(CMDECHO) diff -b -w sepDictLibs.ref sepDictLibs.log


sepDictLibs.log: run.C libmaster.$(DllSuf) libslave1.$(DllSuf) libslave2.$(DllSuf) 
	$(CMDECHO)root.exe -q -l -b run.C 2>&1 | grep -v tagnum | $(RemoveLeadingDirs) | sed -e 's?FILE:[A-Za-z]:.*[/\]?FILE:?' > sepDictLibs.log

all: sepDictLibs

libmaster.$(DllSuf): MyClass.$(ObjSuf) MasterDict.$(ObjSuf)
	$(BuildFromObjs)

libslave1.$(DllSuf):  Slave1Dict.$(ObjSuf)
	$(BuildFromObjs)

libslave2.$(DllSuf):  Slave2Dict.$(ObjSuf)
	$(BuildFromObjs)
    

MasterDict.C: MyClass.hh MasterLinkDef.hh
Slave1Dict.C: MyClass.hh Slave1LinkDef.hh
Slave2Dict.C: MyClass.hh Slave2LinkDef.hh

%Dict.C:
	$(CMDECHO)rootcint -f $@ -c $^ 

%Dict.$(ObjSuf): %Dict.C	
	$(CMDECHO)$(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@ > $@_build.log 2>&1

clean_sepdict:
	$(CMDECHO)rm -f *.$(ObjSuf) *.$(DllSuf) *Dict*
