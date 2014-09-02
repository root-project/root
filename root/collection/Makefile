# %ROOTTEST-duration%:0
TEST_TARGETS += tbits_test TExMap_test TMap_test CATTreeClear

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog tbits TExMap TMap ca.root

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
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

execHistArray.success: EXTRA_DIFFOPTS += -w
