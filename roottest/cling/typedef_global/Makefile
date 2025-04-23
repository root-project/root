TEST_TARGETS += scopeTest scopeTest2
CLEAN_TARGETS += $(ALL_LIBRARIES) output? *Dict* *.log *_timestamp

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

scopeTest : scopeTest_timestamp

scopeTest_timestamp : run.C myclbad.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b run.C 2>&1 | grep -v Processing | $(RemoveLeadingDirs) | sed -e s/myclbad.so/mycl.so/  -e s/10/12/ -e s/::ConstLink/ConstLink/ > output1 ; diff -wb output.ref-$(CINT_VERSION) output1 && touch $@

myclbad.$(DllSuf) : badDict.$(ObjSuf)
	$(BuildFromObj)

badDict.C : bad.h mycl.C LinkDef.h
	$(CMDECHO) rootcint -f badDict.C bad.h mycl.C LinkDef.h


scopeTest2 : scopeTest2_timestamp

scopeTest2_timestamp : run.C myclgood.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b rungood.C 2>&1 | grep -v Processing | $(RemoveLeadingDirs) |sed -e s/myclgood.so/mycl.so\ / > output2 ; diff -wb output.ref-$(CINT_VERSION) output2 && touch $@

myclgood.$(DllSuf) : goodDict.$(ObjSuf)
	$(BuildFromObj)

goodDict.C : mycl.C LinkDef.h
	$(CMDECHO) rootcint -f goodDict.C mycl.C LinkDef.h

