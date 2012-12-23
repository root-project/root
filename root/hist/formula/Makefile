TEST_TARGETS += constargs_test formio_test recurse_test string

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog temp.root constargs formio recurse string

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

constargs_test : constargs

constargs : constargs.clog
	$(TestDiffCintSpecific) && touch $@

ifneq ($(ClingWorkAroundMissingAutoLoading),)
formio.log: CALLROOTEXE += -e 'gSystem->Load("libHist");gSystem->Load("libRIO");gSystem->Load("libGpad");'
endif

formio_test : formio

formio : formio.log
	$(TestDiff) && touch $@

ifneq ($(ClingWorkAroundMissingAutoLoading),)
execGetExp.log assertTernary.elog string.log recurse.log: CALLROOTEXE += -e 'gSystem->Load("libHist");gSystem->Load("libRIO");'
endif

recurse_test : recurse

recurse : recurse.log
	$(TestDiff) && touch $@


string: string.success

string.success : string.log
	$(SuccessTestDiff) && touch $@

string.success: EXTRA_DIFFOPTS += -w 
