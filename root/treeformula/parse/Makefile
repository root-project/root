# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  *.log

#Set the list of target to make while testing
TEST_TARGETS += testStringFormula testBoolOpt expotest NonFormulaMean

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

testStringFormula.log: testString.C
	$(CMDECHO) root.exe -q -l -b testString.C > testStringFormula.log

testStringFormula: testStringFormula.log
	$(CMDECHO) echo testStringFormula > /dev/null

testBoolOpt.log: testBoolOpt.C
	$(CMDECHO) root.exe -b -l -q testBoolOpt.C > testBoolOpt.log 2>&1

testBoolOpt : testBoolOpt.log
	$(CMDECHO) diff -b testBoolOpt.ref testBoolOpt.log

expotest: expotest.log
	$(TestDiff)

NonFormulaMean:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) !(root.exe -q -b -l NonFormulaMean.C)
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


