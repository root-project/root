# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mytest testLoadingSelector

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testLoadingSelector.log: testSel_C.$(DllSuf)  testSelector_C.$(DllSuf)


removepwd: testLoadingSelector.log
ifeq ($(PLATFORM),win32)
	$(CMDECHO)sed 's|<ACLiC>: unloading [A-z:]*[/\\].*[/\\]|<ACLiC>: unloading |g' testLoadingSelector.log > testLoadingSelector.temp 2>&1
else
	$(CMDECHO)sed 's|<ACLiC>: unloading [/\\].*[/\\]|<ACLiC>: unloading |g' testLoadingSelector.log > testLoadingSelector.temp 2>&1
endif
	$(CMDECHO)mv testLoadingSelector.temp testLoadingSelector.log

# ..... | sed -s/$(CURDIR)//g > testLoadingSelector.log


testLoadingSelector: testLoadingSelector.log removepwd
		  $(TestDiff)

