TEST_TARGETS = SelectorTest

include ../../Rules.mk

SelectorTest:
	root.exe -q -b run.C > selector.log 2>&1
	diff selector.log selector.good.log

