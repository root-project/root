# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += oldFormulaTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk


oldFormulaTest.log:  runoldFormulaTest.C
	$(CMDECHO) root.exe -q -b -l runoldFormulaTest.C > oldFormulaTest.log

oldFormulaTest:oldFormulaTest.log
	$(CMDECHO) diff -b oldFormulaTest.log oldFormulaTest.ref

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


