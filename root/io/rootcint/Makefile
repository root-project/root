# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += mytest rootcintIntTest constMemberTest

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

testDict.cxx: test.h testLinkDef.h
	$(CMDECHO) rootcint -f testDict.cxx -c test.h testLinkDef.h

rootcintIntTest: testDict.$(ObjSuf)

constMemberDict.cxx: constMember.h linkdef.h $(ROOTSYS)/bin/rootcint
	$(CMDECHO) rootcint -f constMemberDict.cxx -c constMember.h linkdef.h 

constMemberTest: constMemberDict.$(ObjSuf)
