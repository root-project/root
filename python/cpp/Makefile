# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_cpptests PyROOT_advancedtests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

ifeq ($(PLATFORM),win32)
# The creation of shadow classes in rootcint does not support
# Microsoft's implementation of virtual function overriding 
# when coming from a virtual base class (It would require the
# Shadow class to also override that virtual function).
PyROOT_cpptests:
	$(WarnFailTest)

PyROOT_advancedtests:
	$(WarnFailTest)

else
PyROOT_cpptests.log: Namespace_C.$(DllSuf) PointerPassing_C.$(DllSuf)

PyROOT_cpptests: PyROOT_cpptests.log PyROOT_advancedtests.log
	$(TestDiff)

PyROOT_advancedtests.log: AdvancedCpp_C.$(DllSuf) Template_C.$(DllSuf)

endif
