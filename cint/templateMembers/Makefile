# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log opdict.* templateMembersDict*

#Set the list of target to make while testing
TEST_TARGETS += mytest templateConstructorTest operatorTest templateMembers

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

templateConstructorTest.log : constructor_C.$(DllSuf) runConstructor.C
	$(CMDECHO) root.exe -b -l -q runConstructor.C > templateConstructorTest.log 2>&1

templateConstructorTest: templateConstructorTest.log
	$(CMDECHO) diff -b templateConstructorTest.log templateConstructorTest.ref

opdict.cxx: operator.C linkdef.h
	$(CMDECHO) rootcint -f $@ -c $^

operatorTest: opdict.$(ObjSuf)



templateMembersDict.cxx: templateMembersClasses.h templateMembersLinkdef.h
	$(CMDECHO) rootcint -v4 -f templateMembersDict.cxx -c templateMembersClasses.h templateMembersLinkdef.h

templateMembers_C.$(DllSuf): templateMembersDict.cxx

templateMembers.log: runtemplateMembers.C  templateMembers_C.$(DllSuf) templateMembersCode_C.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q "runtemplateMembers.C(0)" | grep -v Processing > templateMembers.log

templateMembers: templateMembers.log  templateMembers.ref
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo This failures is due to the odd object life time in CINT.
	$(CMDECHO) diff -b templateMembers.ref templateMembers.log | grep -v ::~ > templateMembers.diff.log && \
		diff templateMembers.diff.log templateMembers.diff.ref
else
	$(TestDiff)
endif

templateMembers.ref:  runtemplateMembers.C  templateMembers_C.$(DllSuf)  templateMembersCode_C.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q "runtemplateMembers.C(1)" | grep -v Processing > templateMembers.ref
