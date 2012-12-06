TEST_TARGETS += constargs_test formio_test recurse_test string

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog temp.root constargs formio recurse string

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

constargs_test : constargs

constargs : constargs.clog
	$(TestDiffCintSpecific) && touch $@


formio_test : formio

formio : formio.log
	$(TestDiff) && touch $@


recurse_test : recurse

recurse : recurse.log
	$(TestDiff) && touch $@


string: string.success

string.success : string.log
	$(SuccessTestDiff) && touch $@

