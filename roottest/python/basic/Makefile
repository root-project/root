# This is a template for all makefile.

.PHONY: DataTypes_C Overloads_C

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog __pycache__

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_basictests PyROOT_overloadtests PyROOT_datatypetest PyROOT_operatortests

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


PyROOT_basictests.log: ArgumentPassingCompiled_C.$(DllSuf)

PyROOT_basictests: PyROOT_basictests.log
	$(TestDiff)

#-----
Overloads_C:  Overloads_C.$(DllSuf)
PyROOT_overloadtest.log: Overloads_C

PyROOT_overloadtests: PyROOT_overloadtests.log
	$(TestDiff)


#-----
DataTypes_C:  DataTypes_C.$(DllSuf)
PyROOT_datatypetest.log: DataTypes_C

PyROOT_datatypetest: PyROOT_datatypetest.log
	$(TestDiff)

PyROOT_operatortests.log: Operators_C.$(DllSuf)

PyROOT_operatortests: PyROOT_operatortests.log
	$(TestDiff)

