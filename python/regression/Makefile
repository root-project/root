# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_regressiontests 
#PyROOT_advancedtests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

PyROOT_regressiontests.log: LoKiNamespace_C.$(DllSuf) Marco_C.$(DllSuf) MuonTileID_C.$(DllSuf) Scott2_C.$(DllSuf)  Scott_C.$(DllSuf) ULongLong_C.$(DllSuf)

PyROOT_regressiontests: PyROOT_regressiontests.log
	$(TestDiff)
