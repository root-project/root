# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += testFormulaFuncCall.log

#Set the list of target to make while testing
TEST_TARGETS += testFormulaFuncCall

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

testFormulaFuncCall.log: func.C testcl.C
	$(CMDECHO) root.exe -q -l -b func.C	> testFormulaFuncCall.log 2>&1

testFormulaFuncCall: testFormulaFuncCall.log
	$(CMDECHO) diff -b testFormulaFuncCall.ref testFormulaFuncCall.log
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


