# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_cpptests PyROOT_advancedtests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

PyROOT_cpptests.log: Namespace_C.$(DllSuf) PointerPassing_C.$(DllSuf)

PyROOT_cpptests: PyROOT_cpptests.log
	$(TestDiff)

PyROOT_advancedtests.log: AdvancedCpp_C.$(DllSuf)

ifeq ($(FAIL),)
PyROOT_advancedtests:
	$(WarnFailTest)
else
        PyROOT_advancedtests: PyROOT_advancedtests.log
	$(TestDiff)
endif
