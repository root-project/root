TEST_TARGETS = SelectorTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


SelectorTest:
	$(CMDECHO) root.exe -l -q -b run.C 2>&1 | grep -v reloading > selector.log
	$(CMDECHO) diff -b selector.log selector.good.log

