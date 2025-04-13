# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += TestTClassGlobalIsA

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


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

TestTClassGlobalIsA: TestTClassGlobalIsA.log
	$(CMDECHO) diff -ub  TestTClassGlobalIsA.ref  TestTClassGlobalIsA.log

TestTClassGlobalIsA.log : TestTClassGlobalIsA_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l TestTClassGlobalIsA.C+ > TestTClassGlobalIsA.log 2>&1 || handleError.sh --log=TestTClassGlobalIsA.log --res=$$?

