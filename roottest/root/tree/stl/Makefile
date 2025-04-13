# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog vecptr.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += mixingInterp mixingCompiled mixingGood vstr

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

mixing.clog: runmixing_C.$(DllSuf)

mixing.log: MyClass_h.$(DllSuf)

mixingGoog.clog: runmixingGood_C.$(DllSuf)

mixingCompiled: mixing.clog
	$(TestDiff)

ifneq ($(ClingWorkAroundTypeinfoToTypename),)
mixingInterp: CINT_VERSION := NoTypeinfo
endif

mixingInterp: mixing.log
	$(TestDiffCintSpecific)

mixingGood: mixingGood.clog
	$(TestDiff)

vstr: vstr.clog
	$(TestDiff)
