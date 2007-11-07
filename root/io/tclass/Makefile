# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += TestTClassGlobalIsA

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

ifeq ($(FAIL)$(PLATFORM),win32)
TestTClassGlobalIsA:
	$(WarnFailTest)
else
TestTClassGlobalIsA: TestTClassGlobalIsA.log
	$(CMDECHO) diff -ub  TestTClassGlobalIsA.ref  TestTClassGlobalIsA.log
endif

TestTClassGlobalIsA.log : TestTClassGlobalIsA_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l TestTClassGlobalIsA.C+ > TestTClassGlobalIsA.log 
