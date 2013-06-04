# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += PyClassTest PyAPITest PyROOT_cinttests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


PyClassTest.log: runPyClassTest.C runPyAPITest.C

ifeq ($(FAIL),)
PyClassTest: 
	$(WarnFailTest)
PyAPITest: 
	$(WarnFailTest)
else
PyClassTest: PyClassTest.log
	$(TestDiff)
PyAPITest: PyAPITest.log
	$(TestDiff)
endif

PyROOT_cinttests: PyROOT_cinttests.log
	$(TestDiff)
