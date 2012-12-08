# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.out *.so *.d $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += stringPassingTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

stringPassingTest: passingcint.out passing.out
ifeq ($(ClingWorkAroundJITandInline),)
		  $(CMDECHO)diff -b passingcint.out passingcint.ref
		  $(CMDECHO)diff -b passing.out passing.ref
endif

passingcint.out : t01.C runcint.C
		  $(CMDECHO)$(CALLROOTEXE) -q -b -l runcint.C > passingcint.out 2>buildcint.log

passing.out : t01.C run.C t01_C.$(DllSuf)
		  $(CMDECHO)$(CALLROOTEXE) -q -b -l run.C > passing.out 2>build.log

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

