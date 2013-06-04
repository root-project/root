
TEST_TARGETS += feature
CLEAN_TARGETS += $(ALL_LIBRARIES) run.out *.build.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

ClassTrick_C.$(DllSuf): ClassTrick.C Class_C.$(DllSuf)
	$(BuildWithLib)

run.out: Run.C ClassTrick_C.$(DllSuf)  Class_C.$(DllSuf) typeidtest_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b Run.C >run.out 2>&1

ifeq ($(FAIL)$(PLATFORM),win32)
feature:
else
feature: run.out
endif
