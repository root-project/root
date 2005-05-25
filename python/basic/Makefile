# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += importErrorHandling PyROOT_basictests PyROOT_overloadtests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


PyROOT_basictests: PyROOT_basictests.log
	$(TestDiff)

importErrorHandling: importErrorHandling.log
	$(TestDiff)

PyROOT_overloadtests.log: Overloads_C.$(DllSuf)

PyROOT_overloadtests: PyROOT_overloadtests.log
	$(TestDiff)

PyROOT_datatypetests.log: DataType_C.$(DllSuf)

PyROOT_datatypetests: PyROOT_datatypetests.log
	$(TestDiff)
