# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += longArgTest overloadResolution


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

ifeq ($(FAIL),)
longArgTest:
	$(WarnFailTest)
else
longArgTest: longArgTest.log
	$(TestDiff)
endif

overloadResolution.log: TheClass_h.$(DllSuf)

overloadResolution: overloadResolution.log
	$(TestDiff)
