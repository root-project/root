TEST_TARGETS += sepDictLibs
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.$(ObjSuf) *.$(DllSuf) *Dict*.h *Dict*.C run.log *.build.log

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

ROOTCFLAGS   := $(CXXFLAGS) -DUSE_ROOT
ROOTLIBS     += -lGui

sepDictLibs: run.log
	$(TestDiffCintSpecific)

run.log: run.C libmaster.$(DllSuf) libslave1.$(DllSuf) libslave2.$(DllSuf) run.good.output 
	$(CMDECHO)$(CALLROOTEXE) -q -l -b run.C 2>&1 | grep -v tagnum | $(RemoveLeadingDirs) | $(RemoveSizes) | sed -e 's?FILE:[A-Za-z]:.*[/\]?FILE:?' | sed -e 's?FILE:/.*/?FILE:?' -e '/List of member variables/,/List of member functions/d' > run.log

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
	$(CMDECHO)rootcint -f $@ $^ 

%Dict.$(ObjSuf): %Dict.C	
	$(CMDECHO) $(CXX) $(CXXFLAGS) -DCINTOBJECT -I. $(ROOTCFLAGS) -c $< -o $@

clean_local:
	$(CMDECHO) rm -f *.$(ObjSuf) *.$(DllSuf) *Dict*
