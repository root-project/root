TEST_TARGETS = SelectorTest SelectorCintTest
CLEAN_TARGETS += *.log $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

selector.log: run.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b run.C 2>&1 | grep -v reloading > selector.log

execWithAbort.log: execWithAbort.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b execWithAbort.C 2>&1 | grep -v reloading > execWithAbort.log

electorTest: selector.log
	$(CMDECHO) diff -b selector.ref selector.log

SelectorCintTest.log: sel01_C.$(DllSuf) runSelectorCintTest.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b runSelectorCintTest.C 2>&1 | $(RemoveDirs) > $@ 

SelectorCintTest: SelectorCintTest.log
	$(TestDiffCintSpecific)

execGetSelector.log: emptysel_C.$(DllSuf)
