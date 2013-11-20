CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execStaticConst

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# This is there to let cling know the symbols within the lib
# it can be removed once the feature is added

CALLROOTEXE += -e "gSystem->Load(\"libOdin_dictrflx.so\") "

execStaticConst.log: libOdin_dictrflx.$(DllSuf)
