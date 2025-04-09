TEST_TARGETS = SelectorTest SelectorCintTest
CLEAN_TARGETS += *.log $(ALL_LIBRARIES) lhef*_gen.[Ch]

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

selector.log: run.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b run.C 2>&1 | grep -v reloading > selector.log

ifeq ($(ClingWorkAroundUnloadingVTABLES),)
execWithAbort.log: execWithAbort.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b execWithAbort.C 2>&1 | grep -v reloading > execWithAbort.log
else
execWithAbort.log:
	$(CMDECHO) cp execWithAbort.ref execWithAbort.log
endif

electorTest: selector.log
	$(CMDECHO) diff -b selector.ref selector.log

SelectorCintTest.log: sel01_C.$(DllSuf) runSelectorCintTest.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b runSelectorCintTest.C 2>&1 | $(RemoveDirs) > $@ 

SelectorCintTest: SelectorCintTest.log
	$(TestDiffCintSpecific)

execGetSelector.log: emptysel_C.$(DllSuf)
