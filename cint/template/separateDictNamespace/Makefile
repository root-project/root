ARCH          = $(shell root-config --arch)
PLATFORM      = $(ARCH)

TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  *.$(ObjSuf) *.$(DllSuf) *Dict* run.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


ROOTCFLAGS   := $(shell root-config --cflags) -DUSE_ROOT
ROOTLIBS     := $(patsubst -lCore, -lThread -lCore, $(shell root-config --libs))
ROOTLIBS     += -lGui
ROOTGLIBS    := $(shell root-config --glibs)

sepDictLibs: run.log
	$(CMDECHO) diff -b -w run.good.output run.log

run.log: run.C libmaster.$(DllSuf) libslave1.$(DllSuf) libslave2.$(DllSuf) 
	$(CMDECHO)root.exe -q -l -b run.C 2>&1 | grep -v tagnum | $(RemoveLeadingDirs) | sed -e 's?FILE:[A-Za-z]:.*[/\]?FILE:?' > run.log

all: sepDictLibs

libmaster.$(DllSuf): Master.$(ObjSuf) MasterDict.$(ObjSuf)
	$(BuildFromObjs)

libslave1.$(DllSuf): Slave1.$(ObjSuf)  Slave1Dict.$(ObjSuf)
	$(BuildFromObjs)

libslave3a.$(DllSuf): Slave3a.$(ObjSuf)  Slave3aDict.$(ObjSuf)
	$(BuildFromObjs)

libslave2.$(DllSuf): Slave2.$(ObjSuf)  Slave2Dict.$(ObjSuf)
	$(BuildFromObjs)

libslave3.$(DllSuf): Slave3.$(ObjSuf)  Slave3Dict.$(ObjSuf)
	$(BuildFromObjs)


MasterDict.C: Master.hh MasterLinkDef.hh
Slave1Dict.C: Slave1.hh Master.hh Slave1LinkDef.hh
Slave2Dict.C: Master.hh Slave2.hh Slave2LinkDef.hh

%Dict.C:
	$(CMDECHO)rootcint -f $@ -c $^ 

%Dict.$(ObjSuf): %Dict.C	
	$(CMDECHO) $(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@

clean_local:
	$(CMDECHO) rm -f *.$(ObjSuf) *.$(DllSuf) *Dict*
