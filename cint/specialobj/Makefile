CLEAN_TARGETS += *.log problem.root $(ALL_LIBRARIES)
TEST_TARGETS += stlProxies argtwice

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


argtwice:
	@root.exe -b -l -q 'argeval.C' | grep -v 'root[0]' > argeval.log

tmp.root: AthIndex_h.$(DllSuf)  T0Result_h.$(DllSuf)  calib_h.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q write.C > tmp_root.log 2>&1

T0Result_h.$(DllSuf): T0Result.h AthIndex_h.$(DllSuf)
	$(BuildWithLib)

stlProxies.log: tmp.root AthIndex_h.$(DllSuf)  T0Result_h.$(DllSuf)  calib_h.$(DllSuf)

stlProxies: stlProxies.log
	$(TestDiff)
