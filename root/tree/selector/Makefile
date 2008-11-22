TEST_TARGETS = SelectorTest SelectorCintTest
CLEAN_TARGETS += *.log $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

selector.log: run.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -l -q -b run.C 2>&1 | grep -v reloading > selector.log

SelectorTest: selector.log
	$(CMDECHO) diff -b selector.ref selector.log

SelectorCintTest.log: sel01_C.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b runSelectorCintTest.C 2>&1 | $(RemoveDirs) > $@ 

SelectorCintTest: SelectorCintTest.log
	$(TestDiff)

