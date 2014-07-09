CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execClassInNamespace

ifeq ($(strip $(ROOTTEST_HOME)),)
        export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

execClassInNamespace.log: libclassInNamespace_dictrflx.$(DllSuf)  
