# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  testStringFormula.log

#Set the list of target to make while testing
TEST_TARGETS += testStringFormula

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

testStringFormula.log: testString.C
	$(CMDECHO) root.exe -q -l -b testString.C > testStringFormula.log

testStringFormula: testStringFormula.log
	$(CMDECHO) echo testStringFormula > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


