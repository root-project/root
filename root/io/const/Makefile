# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log *.root $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += constTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


constTest.log: constTest.C
	$(CMDECHO) root -q -b -l constTest.C+ > constTest.log 

constTest: constTest.log
	$(CMDECHO) diff -b constTest.log constTest.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


