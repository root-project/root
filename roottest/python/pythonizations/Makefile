# This is a template for all makefile.

.PHONY: Pythonizables_C

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog __pycache__ foo.root

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_pythonizationtest PyROOT_smartptrtest

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


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


Pythonizables_C:  Pythonizables_C.$(DllSuf)
PyROOT_pythonizationtest.log: Pythonizables_C.$(DllSuf)

PyROOT_pythonizationtest: PyROOT_pythonizationtest.log
	$(TestDiff)

SmartPtr_C:  SmartPtr_C.$(DllSuf)
PyROOT_smartptrtest.log: SmartPtr_C.$(DllSuf)

PyROOT_smartptrtest: PyROOT_smartptrtest.log
	$(TestDiff)
