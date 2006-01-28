# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) scopeDict.cxx

#Set the list of target to make while testing
TEST_TARGETS += operatorLessThanTemplate testTable scopeTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testTable: testTable.log
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo the problem exist when BYTECODE is \#defined in TTable.cxx
else
	$(TestDiff)
endif

operatorLessThanTemplate.log: runoperatorLessThanTemplate.C operatorLessThanTemplate_C.$(DllSuf)
	$(CMDECHO) root.exe -q -l -b  runoperatorLessThanTemplate.C | $(RemoveLeadingDirs) | grep -v -e tagnum -e size= > $@ 2>&1
	@echo "Warning: operatorLessThanTemplate has some known defect (needs to modify the file to see them)"

operatorLessThanTemplate:  operatorLessThanTemplate.log
	$(TestDiff)

scopeDict.cxx: scopeProblem.C
	$(CMDECHO) rootcint -f scopeDict.cxx -c scopeProblem.C linkdef.h

scopeTest: scopeDict.cxx

operatorNeg: operatorNeg.log
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(TestDiff)
endif
