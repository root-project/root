# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mytest testLoadingSelector

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testLoadingSelector.log: testSel_C.$(DllSuf)  testSelector_C.$(DllSuf)


testLoadingSelector: testLoadingSelector.log
		  $(TestDiff)

