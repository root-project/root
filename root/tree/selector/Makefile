TEST_TARGETS = SelectorTest

include ../../Rules.mk

SelectorTest:
	$(CMDECHO) root.exe -l -q -b run.C 2>&1 | grep -v reloading > selector.log
	$(CMDECHO) diff selector.log selector.good.log

