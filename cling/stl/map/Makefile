# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += dict* *.o $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mapTest badOperatorTest stringMapTest badstringMapTest

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

mapTest: dict.o testing.o
	$(CMDECHO) echo mytest > /dev/null

stringMap.log :stringMap.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l stringMap.cxx  > stringMap.log 2>&1 ||  handleError.sh --result=$$? --log=stringMap.log --test=stringMapTest

stringMap.log: stringMapLoad_cxx.$(DllSuf)

stringMapTest: stringMap.log
	$(TestDiff)

badstringMap.log : stringMapLoad_cxx.$(DllSuf) badstringMap.cxx MyClass_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l badstringMap.cxx  > badstringMap.log 2>&1 || handleError.sh --result=$$? --log=badstringMap.log --test=badstringMapTest

badstringMapTest: badstringMap.log
	$(TestDiff)

badOperator.log: badOperator.C MyOpClass_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q badOperator.C > badOperator.log 2>&1 || handleError.sh --result=$$? --log=badOperator.log --test=badOperatorTest

badOperatorTest: badOperator.log
	$(TestDiff)

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

dict.C: testing.C
	$(CMDECHO) rootcint -f dict.C testing.C LinkDef.h


