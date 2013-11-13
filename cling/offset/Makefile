TEST_TARGETS += vbase-i vbase-c
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


vbase-i: vbase.log
	$(TestDiff)

vbase-c: vbase.clog
	$(TestDiff)
