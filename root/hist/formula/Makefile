TEST_TARGETS += constargs_test formio_test recurse_test string

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog temp.root constargs formio recurse string

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
