CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *pcm *o *rflx.* dummy* al.rootmap ofile.root

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# check sanity of pcm name
execmangledTest.log: libmangledTest_lhapdf-5.9.1_dictrflx.$(DllSuf)
# escape quotes test
execwriteFirstRootmap.log: execwriteFirstRootmap.C
execwriteTree.log: libMyClass_v1_dictrflx.$(DllSuf) execwriteFirstRootmap.log
execreadTree.log: execwriteTree.log libMyClass_v2_dictrflx.$(DllSuf)

execTypedefSelection.log: libns_dictrflx.$(DllSuf)
execTypedefSelection.log: libtypedefSelection_dictrflx.$(DllSuf)
execstlPatternSelection.log: libstlPatternSelection_dictrflx.$(DllSuf)
execCheckComplicatedAllocator.log: libcomplicatedAllocator_dictrflx.$(DllSuf)
execns.log: libns_dictrflx.$(DllSuf)
execAttributesCheck.log: libattributesCheck_dictrflx.$(DllSuf)

