# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += addTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

addTest:
	$(CMDECHO) root.exe -b -q -l run.C > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


