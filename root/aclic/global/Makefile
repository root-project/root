TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o $(ALL_LIBRARIES) *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

GlobalAclicTest: id_cxx.so idDict.cxx

id_cxx.so: id.cxx buildId.C
	$(CMDECHO) root.exe -l -b -q buildId.C >& id.log

idDict.cxx: id.cxx
	$(CMDECHO) rootcint -f idDict.cxx -c id.cxx LinkDef.h
