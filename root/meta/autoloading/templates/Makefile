CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* *rootmap $(ALL_LIBRARIES)
TEST_TARGETS += execTemplateAutoloading

ifeq ($(strip $(ROOTTEST_HOME)),)
        export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

GENREFLEXCXXFLAGS += --rootmap libclasses_dictrflx.rootmap --rootmap-lib libclasses_dictrflx.$(DllSuf)

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

execTemplateAutoloading.log: libclasses_dictrflx.$(DllSuf)
