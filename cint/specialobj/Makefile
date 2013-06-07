CLEAN_TARGETS += *.log problem.root $(ALL_LIBRARIES) tmp.root
TEST_TARGETS += stlProxies argtwice

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


argtwice:
	@$(CALLROOTEXE) -b -l -q 'argeval.C' | grep -v 'root[0]' > argeval.log

tmp.root: AthIndex_h.$(DllSuf)  T0Result_h.$(DllSuf)  calib_h.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q write.C > tmp_root.log 2>&1 ||  handleError.sh --result=$$? --log=tmp_root.log --test=tmp.root 

T0Result_h.$(DllSuf): T0Result.h AthIndex_h.$(DllSuf)
	$(BuildWithLib)

stlProxies.log: tmp.root AthIndex_h.$(DllSuf)  T0Result_h.$(DllSuf)  calib_h.$(DllSuf)

stlProxies: stlProxies.log
	$(TestDiff)
