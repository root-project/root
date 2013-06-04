# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog *.root

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_ttreetests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

PyROOT_ttreetests.log: TTreeTypes_C.$(DllSuf)

PyROOT_ttreetests: PyROOT_ttreetests.log
	$(TestDiff)
