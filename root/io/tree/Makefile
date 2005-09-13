# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root copy.root *.so *.out *.d test2.root

#Set the list of target to make while testing
TEST_TARGETS += missingClass reftest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


missingClass: read.out
	$(CMDECHO) diff -bw read.ref read.out


read.out: test.root copy.root
	$(CMDECHO) root.exe -q -l -b read.C > read.out 2>&1 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

test.root: classes.C write.C
		  $(CMDECHO) root.exe -q -l -b write.C > write.out 2>&1

copy.root: classes.C copy.C test.root
		  $(CMDECHO) root.exe -q -l -b copy.C > copy.out 2>&1

test2.root: tref_test.C
		  $(CMDECHO) root.exe -q -l -b tref_test.C > tref_test.log

tref_test_rd.log: tref_test_rd.C test2.root
		  $(CMDECHO) root.exe -q -l -b tref_test_rd.C > tref_test_rd.log

reftest: tref_test_rd.log
		  $(TestDiff)