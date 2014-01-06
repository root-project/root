CLEAN_TARGETS += *.log *.so *.d *pcm *rflx* dummy*.C *.root *success
TEST_TARGETS += execbasic

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

execstringIO.log: libstringIO_allClasses_dictrflx.$(DllSuf)

