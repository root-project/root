# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += dict* *.o $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mapTest badOperatorTest stringMapTest badstringMapTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

mapTest: dict.o testing.o
	$(CMDECHO) echo mytest > /dev/null

stringMap.log :stringMap.cxx
	$(CMDECHO) $(CALLROOTEXE) -q -b -l stringMap.cxx  > stringMap.log 2>&1

stringMap.log: stringMapLoad_cxx.$(DllSuf)

stringMapTest: stringMap.log
	$(CMDECHO) diff -b stringMap.log stringMap.ref

badstringMap.log : stringMapLoad_cxx.$(DllSuf) badstringMap.cxx MyClass_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l badstringMap.cxx  > badstringMap.log 2>&1

badstringMapTest: badstringMap.log
	$(CMDECHO) diff -b badstringMap.log badstringMap.ref

badOperator.log: badOperator.C MyOpClass_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q badOperator.C > badOperator.log 2>&1

badOperatorTest: badOperator.log
	$(CMDECHO) diff -b badOperator.log badOperator.ref

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

dict.C: testing.C
	$(CMDECHO) rootcint -f dict.C -c testing.C LinkDef.h


