CLEAN_TARGETS += *.so $(ALL_LIBRARIES) *.clog *.log *.tmp
TEST_TARGETS += IgnoreTObject

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

IgnoreTObject : IgnoreTObject.clog
	$(TestDiff)

