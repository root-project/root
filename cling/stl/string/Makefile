# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.out *.so *.d $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += stringPassingTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

stringPassingTest: passingcint.out passing.out
ifeq ($(ClingWorkAroundJITandInline),)
ifeq ($(SUMMARY),)
		$(CMDECHO)diff -b passingcint.out passingcint.ref 
		(CMDECHO)diff -b passing.out passing.ref
else
		  $(CMDECHO)diff -b passingcint.out passingcint.ref > $(SUMMARY).passingcint.diff.log || handleError.sh --cmd=diff --result=$$? --log=$(SUMMARY).passingcint.diff.log --test=stringPassingTest
		  $(CMDECHO)diff -b passing.out passing.ref > $(SUMMARY).passing.diff.log  || handleError.sh --cmd=diff --result=$$? --log=$(SUMMARY).passing.diff.log --test=stringPassingTest
endif
endif

passingcint.out : t01.C runcint.C
		  $(CMDECHO)$(CALLROOTEXE) -q -b -l runcint.C > passingcint.out 2>buildcint.log|| handleError.sh --result=$$? --log=passingcint.out --test=stringPassingTest 

passing.out : t01.C run.C t01_C.$(DllSuf)
		  $(CMDECHO)$(CALLROOTEXE) -q -b -l run.C > passing.out 2>build.log || handleError.sh --result=$$? --log=passing.out --test=stringPassingTest

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

