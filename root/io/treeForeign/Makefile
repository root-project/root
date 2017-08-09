# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root output create.out def_C* *.log

#Set the list of target to make while testing
TEST_TARGETS += testForeignDraw

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


test.root: def.C def.h $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q def.C+ > create.log 2>&1

testForeignDraw.log: Run.C test.root 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q Run.C 2>&1 | tee fulloutput.log | grep -v 'Total =' > testForeignDraw.log

ifeq ($(lz4),)
    testForeignDraw: testForeignDraw.log
	$(CMDECHO) diff -ubw testForeignDrawLZ4.ref testForeignDraw.log
else
    testForeignDraw: testForeignDraw.log
	$(CMDECHO) diff -ubw testForeignDraw.ref testForeignDraw.log
endif

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


