CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execEnumsTest

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

execEnumsTest.log: libenumsTestClasses_dictrflx.$(DllSuf)
execenumsInNamespaces.log: libenumsInNamespaces_dictrflx.$(DllSuf)
execenumsInNamespaces2.log: libenumsInNamespaces2_dictrflx.$(DllSuf)
execduplicateEnums.log: libduplicateEnums_dictrflx.$(DllSuf)
execQualifiedNames.log: libqualifiedNames_dictrflx.$(DllSuf)
execfclassVal.log: libfclassVal_dictrflx.$(DllSuf)
execenumsInNamespaces.log: libfclassVal_dictrflx.$(DllSuf)
