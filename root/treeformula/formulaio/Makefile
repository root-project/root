# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += oldFormulaTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


oldFormulaTest.log:  runoldFormulaTest.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runoldFormulaTest.C > oldFormulaTest.log

oldFormulaTest:oldFormulaTest.log
	$(CMDECHO) diff -bw oldFormulaTest.log oldFormulaTest.ref

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


