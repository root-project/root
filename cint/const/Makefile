TEST_TARGETS += constTest constCharTest
CLEAN_TARGET += $(ALL_LIBRARIES) output

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


constTest: const.log
	$(CMDECHO) diff const.log good.output

const.log : const.C const_C.$(DllSuf)
	$(CMDECHO) echo "gROOT->ProcessLine(\".L const.C\");run()" | root.exe -b -l  2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' > const.log  2>&1
	$(CMDECHO) echo "gROOT->ProcessLine(\".L const.C+\");run()" | root.exe -b -l  2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' >> const.log 2>&1 

constCharTest: 
	$(CMDECHO) root.exe -b -q -l ../../root/build.C\(\"constissue.h\"\) > constCharTest.log
