# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += testFormulaFuncCall.log prova.root *.log $(ALL_LIBRARIES) \
        test.root

#Set the list of target to make while testing
TEST_TARGETS += fullload staticFunc testFormulaFuncCall nestedFunc nestedHists Timestamp subform

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


testFormulaFuncCall.log: func.C testcl.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b func.C	> testFormulaFuncCall.log 2>&1

testFormulaFuncCall: testFormulaFuncCall.log
	$(CMDECHO) diff -b testFormulaFuncCall.ref testFormulaFuncCall.log
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

prova.root: lista_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l lista.C+ > prova_root.log

nestedHists.log: prova.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runnestedHists.C  2>&1 | grep -v 'Warning in <TClass::Init>: no dictionary for class' > nestedHists.log

nestedHists: nestedHists.log 
	$(TestDiff)

nestedFunc: nestedFunc.log
	$(TestDiff)

staticFunc.log: MyClass_cxx.$(DllSuf)

staticFunc: staticFunc.log
	$(TestDiff)

all_C.$(DllSuf): A.h B.h ALinkDef.h BLinkDef.h

test.root: write.C all_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runwrite.C > write.log 2>&1

fullload.log: test.root

fullload: fullload.log
ifeq ($(ClingWorkAroundCallfuncAndInline)$(ClingWorkAroundCallfuncAndReturnByValue),)
	$(TestDiff)
else
	$(TestDiffCintSpecific)
endif

Timestamp: Timestamp.log
ifneq ($(ClingWorkAroundCallfuncAndInline),)
	@true
else
	$(TestDiff)
endif

subform: subform.log
	$(TestDiff)

