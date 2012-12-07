# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += constRetrunTest barna

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

constRetrunTest:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) $(CALLROOTEXE) -q -b -l run.C > run.log
endif

barna.log: barna.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l barna.C > barna.log 2>&1

barna: barna.log
	$(TestDiffW)
