# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += defaultParams longArgTest overloadResolution staticfunc


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
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

defaultParams.log: Params_h.$(DllSuf)

# This will skip the test (if there is a known failure)
ifeq ($(FAIL),)
defaultParams:
	$(WarnFailTest)
else
defaultParams: defaultParams.log
	$(TestDiff)
endif

staticfunc.log: staticfunc.C
	$(CMDECHO) root.exe -b -q -l runstaticfunc.C 2>&1 | $(RemoveDirs) -e 's?[(]20[)]?:20:?' > staticfunc.log

staticfunc: staticfunc.log
	$(TestDiff)
