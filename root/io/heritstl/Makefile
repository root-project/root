# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += minostest 

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

minostest.log: minostest_C.$(DllSuf) classes_C.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q runminostest.C 2>&1 | sed -e 's/offset= [0-9]*/offset= \[deleted from log\]/g' > minostest.log

minostest: minostest.log
		  $(TestDiff)
