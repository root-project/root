TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o $(ALL_LIBRARIES) *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

GlobalAclicTest: id_cxx.$(DllSuf) idDict.cxx

id_cxx.$(DllSuf): id.cxx buildId.C
#	$(CMDECHO) $(CALLROOTEXE) -l -b -q buildId.C > id.log 2&1

idDict.cxx: id.cxx
	$(CMDECHO) rootcint -f idDict.cxx -c id.cxx LinkDef.h
