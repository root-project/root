CLEAN_TARGETS += 
TEST_TARGETS += argtwice

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


argtwice:
	@(echo '.x argeval.C' | root -l -b)