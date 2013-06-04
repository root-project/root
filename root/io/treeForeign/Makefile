# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root output create.out def_C* *.log

#Set the list of target to make while testing
TEST_TARGETS += testForeignDraw

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


test.root: def.C def.h $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q def.C+ > create.log 2>&1

testForeignDraw.log: Run.C test.root 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q Run.C 2>&1 | tee fulloutput.log | grep -v 'Total =' > testForeignDraw.log

testForeignDraw: testForeignDraw.log 
	$(CMDECHO) diff -ubw testForeignDraw.ref testForeignDraw.log

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


