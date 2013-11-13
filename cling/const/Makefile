TEST_TARGETS += constTest constCharTest
CLEAN_TARGETS += $(ALL_LIBRARIES) output *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


constTest: const.log
	$(CMDECHO) diff -bw good.output const.log

const.log : runConst.C runConst_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q runConst.C  2>&1 | grep -v 'Info in <T.*System::ACLiC>' > const.log  2>&1
	$(CMDECHO) $(CALLROOTEXE) -b -l -q runConst.C+ 2>&1 | grep -v 'Info in <T.*System::ACLiC>' >> const.log 2>&1 

constCharTest: 
	$(CMDECHO) $(CALLROOTEXE) -b -q -l ../../root/build.C\(\"constissue.h\"\) > constCharTest.log 2>&1 
