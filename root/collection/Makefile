TEST_TARGETS += tbits_test TExMap_test TMap_test

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog tbits TExMap TMap

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

tbits_test : tbits

tbits : tbits.log
	$(TestDiff) && touch $@


TExMap_test : TExMap

TExMap : TExMap.clog
	$(TestDiff) && touch $@


TMap_test : TMap

TMap : TMap.clog
	$(TestDiff) && touch $@
