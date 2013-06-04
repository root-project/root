
TEST_TARGETS += sepDictLibs
CLEAN_TARGETS += $(ALL_LIBRARIES) *.$(ObjSuf) *.$(DllSuf) *Dict.[Ch] *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

ROOTCFLAGS   := $(CXXFLAGS) -DUSE_ROOT
ROOTLIBS     += -lGui

sepDictLibs: sepDictLibs.log
	$(TestDiffCintSpecific)
#	$(CMDECHO) diff -b -w sepDictLibs.ref sepDictLibs.log


sepDictLibs.log: run.C libmaster.$(DllSuf) libslave1.$(DllSuf) libslave2.$(DllSuf) sepDictLibs.ref
	$(CMDECHO)$(CALLROOTEXE) -q -l -b run.C 2>&1 | grep -v tagnum | $(RemoveLeadingDirs) | sed -e 's?FILE:[A-Za-z]:.*[/\]?FILE:?' -e 's?FILE:/.*/?FILE:?' | $(RemoveSizes) > sepDictLibs.log

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
