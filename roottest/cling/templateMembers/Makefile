# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log opdict.* templateMembersDict* templateMembers.ref templateMembersNoMem.ref

#Set the list of target to make while testing
TEST_TARGETS += duplicate operatorEqual mytest templateConstructorTest operatorTest 
# templateMembers

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

templateConstructorTest.log : constructor_C.$(DllSuf) runConstructor.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q runConstructor.C > templateConstructorTest.log 2>> templateConstructorTest.build.log

templateConstructorTest: templateConstructorTest.log
	$(CMDECHO) diff -wb templateConstructorTest.log templateConstructorTest.ref

opdict.cxx: operator.C linkdef.h
	$(CMDECHO) rootcint -f $@ $^ > opdict.build.log 2>&1

operatorTest: opdict.$(ObjSuf)



templateMembersDict.cxx: templateMembersClasses.h templateMembersLinkdef.h
	$(CMDECHO) rootcint -v4 -f templateMembersDict.cxx templateMembersClasses.h templateMembersLinkdef.h 2>>templateMembers.build.log

templateMembers_C.$(DllSuf): templateMembersDict.cxx

templateMembersNoMem.log: templateMembers.log
	$(CMDECHO) grep -v Mem: templateMembers.log > templateMembersNoMem.log

templateMembers.log: runtemplateMembers.C  templateMembers_C.$(DllSuf) templateMembersCode_C.$(DllSuf) templateMembers.ref
	$(CMDECHO) $(CALLROOTEXE) -b -l -q "runtemplateMembers.C(0)" | grep -v -e Processing > templateMembers.log && \
	grep -v Mem: templateMembers.log > templateMembersNoMem.log

templateMembers: templateMembers.log  templateMembers.ref
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO)echo This failures is due to the odd object life time in CINT.
	$(CMDECHO)diff -wb templateMembersNoMem.ref templateMembersNoMem.log 
else
	$(TestDiff)
endif

templateMembers.ref:  runtemplateMembers.C  templateMembers_C.$(DllSuf)  templateMembersCode_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q "runtemplateMembers.C(1)" | grep -v -e Processing -e Mem: > templateMembers.ref && \
	grep -v Mem: templateMembers.ref > templateMembersNoMem.ref

templateMembersCode_C.$(DllSuf): templateMembersCode.C templateMembers_C.$(DllSuf)
	$(BuildWithLib)

operatorEqual.log: operatorEqual_C.$(DllSuf) runoperatorEqual.C
	$(CMDECHO)$(CALLROOTEXE) -q -b -l runoperatorEqual.C | grep -v FILE | grep -v tagnum |  $(RemoveDirs)| $(RemoveLeadingDirs) > operatorEqual.log

operatorEqual: operatorEqual.log
	$(TestDiffCintSpecific)

duplicate.log: duplicate_C.$(DllSuf)

duplicate: duplicate.log
	$(TestDiff)
