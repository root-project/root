# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += testFormulaFuncCall.output

#Set the list of target to make while testing
TEST_TARGETS += testFormulaFuncCall

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

testFormulaFuncCall.output: func.C testcl.C
	$(CMDECHO) root.exe -q -l -b func.C	> testFormulaFuncCall.output 2>&1

testFormulaFuncCall: testFormulaFuncCall.output
	$(CMDECHO) diff testFormulaFuncCall.good testFormulaFuncCall.output
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


