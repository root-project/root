TEST_TARGETS += constTest constCharTest
CLEAN_TARGETS += $(ALL_LIBRARIES) output *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


constTest: const.log
	$(CMDECHO) diff const.log good.output

const.log : runConst.C runConst_C.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q runConst.C  2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' > const.log  2>&1
	$(CMDECHO) root.exe -b -l -q runConst.C+ 2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' >> const.log 2>&1 

constCharTest: 
	$(CMDECHO) root.exe -b -q -l ../../root/build.C\(\"constissue.h\"\) > constCharTest.log
