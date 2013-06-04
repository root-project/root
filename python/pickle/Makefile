# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog *.root *.pcl *.cpcl *.pyc

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_readtests PyROOT_writetests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

PyROOT_writetests.log: PickleTypes_C.$(DllSuf)

PyROOT_writetests: PyROOT_writetests.log
	$(TestDiff)

PyROOT_readtests.log: PickleTypes_C.$(DllSuf) PyROOT_writetests

PyROOT_readtests: PyROOT_readtests.log
	$(TestDiff)
