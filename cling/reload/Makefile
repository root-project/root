# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += reloadvector

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


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

reloadvector: reloadvector.log
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo 'Without the cintdlls reloading <vector> leads to warnings'
else
	$(TestDiff)
endif

