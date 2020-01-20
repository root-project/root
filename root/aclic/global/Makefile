TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o $(ALL_LIBRARIES) *.log

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

GlobalAclicTest: id_cxx.$(DllSuf) idDict.cxx

id_cxx.$(DllSuf): id.cxx buildId.C
#	$(CMDECHO) $(CALLROOTEXE) -l -b -q buildId.C > id.log 2&1

idDict.cxx: id.cxx
	$(CMDECHO) rootcint -f idDict.cxx id.cxx LinkDef.h
