
TEST_TARGETS += feature
CLEAN_TARGETS += $(ALL_LIBRARIES) run.out *.build.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

run.out: Run.C ClassTrick_C.$(DllSuf)  Class_C.$(DllSuf) typeidtest_C.$(DllSuf)
	$(CMDECHO)root -q -l -b Run.C >run.out 2>&1

feature: run.out
