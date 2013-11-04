CLEAN_TARGETS += *.log *so *pcm *rflx* dummy*.C
TEST_TARGETS += execloadLib

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# Create the dict source
execloadLib.log:  libclasses_dictrflx.$(DllSuf)
