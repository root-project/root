TEST_TARGETS += vbase-i vbase-c
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

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


vbase-i: vbase.log
	$(TestDiff)

vbase-c: vbase.clog
	$(TestDiff)
