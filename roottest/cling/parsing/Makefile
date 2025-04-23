# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log $(ALL_LIBRARIES) scopeDict.* scopeDict2.*  TTestClass_h_DEBUGTARGET

#Set the list of target to make while testing
TEST_TARGETS += lessThan operatorLessThanTemplate testTable scopeTest multiScope Readbool ptrconst TTestClass semicolon

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Remember that sometest.log can be made automatically from
# from runsometest.C

ifeq ($(shell root-config --has-table),yes)
ifeq (DISABLED!,)
testTable: testTable.log
endif
testTable:
	@echo '$@ DISABLED until transactions are fixed!'
else
testTable:
endif
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo the problem exist when BYTECODE is \#defined in TTable.cxx
else
	$(TestDiff)
endif

operatorLessThanTemplate.log: runoperatorLessThanTemplate.C operatorLessThanTemplate_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b  runoperatorLessThanTemplate.C | $(RemoveLeadingDirs) | grep -v -e tagnum -e size= > $@ 2>&1
ifneq ($(FAIL),)
	@echo "Warning: Test operatorLessThanTemplate has some known defect (needs to modify the file to see them)"
endif

operatorLessThanTemplate:  operatorLessThanTemplate.log
ifeq ($(ClingWorkAroundNoDotNamespace),)
	$(TestDiff)
endif

scopeDict.cxx: scopeProblem.C
	$(CMDECHO) rootcint -f scopeDict.cxx scopeProblem.C linkdef.h

scopeDict2.cxx: multiScope.C
	$(CMDECHO) rootcint -f $@ $< linkdef.h

multiScope: scopeDict2.o

lessThan: lessThan.log
	$(TestDiff)

scopeTest: scopeDict.cxx

operatorNeg: operatorNeg.log
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(TestDiff)
endif

Readbool: Readbool.log
	$(TestDiffW)

ptrconst.clog: ptrconst_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l ptrconst.C+ > ptrconst.clog 2>&1

ptrconst: ptrconst.clog
	$(TestDiffW)

TTestClass_h.$(DllSuf):

TTestClass.log: TTestClass_h.$(DllSuf)

TTestClass: TTestClass.log
	$(TestDiff)

semicolon.log: semicolon.C semicolon.ref
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'semicolon.C("abc ; xyz")' > semicolon.log 2>&1 || handleError.sh --result=$$? --log=$@ --test=$@

semicolon: semicolon.log
	$(TestDiff)
