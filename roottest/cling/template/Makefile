#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log $(ALL_LIBRARIES) *.out

#Set the list of target to make while testing
TEST_TARGETS += templateSingleton longlong usingtemplate operator1 simpleSeparateDict forwardDecl constructorTest typenameTest \
	templatefriend

SHELL = /bin/sh

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


typenameTest:
	$(CMDECHO)rootcint -f typenameTestDict.C -D__CINT2__ typenameTest.C linkdef.h 
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO) echo "this is fixed by cling."
else
	$(CMDECHO)rootcint -f typenameTestDict.C typenameTest.C linkdef.h 
endif

operator1:
	$(CMDECHO)rm -f operator1.cint.out operator1.comp.out
	$(CMDECHO) $(CALLROOTEXE) -q -l -b smalltest.C > operator1.cint.out 2>&1
	$(CMDECHO) $(CALLROOTEXE) -q -l -b smalltest.C+ > operator1.comp.out 2>&1
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO)cat operator1.comp.out | grep -v 'not declared' |grep -v 'Info in <ACLiC>:' | diff - operator1.cint.out
endif


simpleSeparateDict: 
	$(CMDECHO)rootcint -f masterDict.C master.h masterLinkDef.h
	$(CMDECHO)rootcint -f slaveDict.C slave.h slaveLinkDef.h

forwardDecl: forwardDict.C

forwardDict.C: forward.C 
	$(CMDECHO)rootcint -f forwardDict.C forward.C linkdef.h


constructorDict.C: constructor.hh 
	$(CMDECHO) rootcint -f constructorDict.C constructor.hh linkdef.h

constructorTest: constructorDict.o

usingtemplate.log: usingtemplate_C.$(DllSuf)

usingtemplate: usingtemplate.log
	$(TestDiff)

longlong.log: longlong_C.$(DllSuf)

longlong: longlong.log
	$(TestDiff)

templatefriend.log : templatefriend_cxx.$(DllSuf)

templatefriend: templatefriend.log
	$(TestDiffW)

templateSingleton.log: Singleton_h.$(DllSuf)

templateSingleton: templateSingleton.log
	$(TestDiff)
