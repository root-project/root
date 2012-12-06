# %ROOTTEST-duration%:0
TEST_TARGETS += tbits_test TExMap_test TMap_test CATTreeClear

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog tbits TExMap TMap ca.root

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

tbits_test : tbits

tbits : tbits.log
ifeq ($(ClingWorkAroundJITandInline),)
	$(TestDiff) && touch $@
endif

TExMap_test : TExMap

TExMap : TExMap.clog
	$(TestDiff) && touch $@


TMap_test : TMap

TMap : TMap.clog
	$(TestDiff) && touch $@

CATTreeClear : CATTreeClear.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
CATTreeClear.log : CALLROOTEXE += -e "gSystem->Load(\"libTree\");"
endif

ifneq ($(ClingWorkAroundErracticValuePrinter),)
execHistArray.success: EXTRA_DIFFOPTS += -w
endif
