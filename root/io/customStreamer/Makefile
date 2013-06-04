# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += customStreamerTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

customStreamerTest.log: header_C.$(DllSuf) run.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l run.C 2>&1 | grep -v 'creating shared library ' > $@

customStreamerTest: customStreamerTest.log
	$(CMDECHO) diff -b customStreamerTest.log customStreamerTest.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


