# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += depth.root create.output

#Set the list of target to make while testing
TEST_TARGETS += depthTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

ifneq ($(FAIL),)
depthTest: depth.root
	$(CMDECHO) root.exe -q -l -b Read.C 
else
depthTest: 
	$(CMDECHO) echo depthTest known to fail
endif

depth.root: Create.C
	$(CMDECHO) root.exe -q -l -b Create.C > create.output

