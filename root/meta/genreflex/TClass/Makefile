CLEAN_TARGETS += *.log *.so *.d *pcm *rflx* dummy*.C
TEST_TARGETS += execbasic

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# Basic test -------------------------------------------------------------------

execbasic.log: libbasic_allClasses_dictrflx.$(DllSuf)

# End of Basic test ------------------------------------------------------------
