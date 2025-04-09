# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log test.root $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += constTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


constTest.log: constTest_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l constTest.C+ > constTest.log 

constTest: constTest.log
	$(CMDECHO) diff -b constTest.log constTest.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


