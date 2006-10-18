CLEAN_TARGETS += *.so *.root $(ALL_LIBRARIES) *.clog *.log *.tmp
TEST_TARGETS += IgnoreTObject

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

IgnoreTObject : IgnoreTObject.clog
	$(TestDiff)

