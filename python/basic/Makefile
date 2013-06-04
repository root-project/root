# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_basictests PyROOT_overloadtests PyROOT_datatypetest PyROOT_operatortests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


PyROOT_basictests.log: ArgumentPassingCompiled_C.$(DllSuf)

PyROOT_basictests: PyROOT_basictests.log
	$(TestDiff)

PyROOT_overloadtests.log: Overloads_C.$(DllSuf)

PyROOT_overloadtests: PyROOT_overloadtests.log
	$(TestDiff)

PyROOT_datatypetest.log: DataTypes_C.$(DllSuf)

PyROOT_datatypetest: PyROOT_datatypetest.log
	$(TestDiff)

PyROOT_operatortests.log: Operators_C.$(DllSuf)

PyROOT_operatortests: PyROOT_operatortests.log
	$(TestDiff)
