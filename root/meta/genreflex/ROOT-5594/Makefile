CLEAN_TARGETS += *.log xr.* dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execdummy

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# Create the dict source
execdummy.log:  AliAODPid_rflx.$(ObjSuf)