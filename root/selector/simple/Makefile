# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mytest testLoadingSelector

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
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


removepwd: 
	$(CMDECHO)sed 's|<ACLiC>: unloading [/\\].*[/\\]|<ACLiC>: unloading |g' testLoadingSelector.log >& testLoadingSelector.temp
	$(CMDECHO)mv testLoadingSelector.temp testLoadingSelector.log

# ..... | sed -s/$(PWD)//g > testLoadingSelector.log


testLoadingSelector: testLoadingSelector.log removepwd
		  $(TestDiff)

