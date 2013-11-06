CLEAN_TARGETS += *.log xr.* dummy* *pcm *o *rflx.*
TEST_TARGETS += root_5626

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# Create the dict source
execdummy.log:  classes_rflx.$(ObjSuf)
