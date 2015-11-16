CLEAN_TARGETS += *.log xr.* dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execdummy

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk

HEADERS = ROOT-7775/BaseInfo.h ROOT-7775/CLASS_DEF.h ROOT-7775/incident.h ROOT-7775/part3.h \
          ROOT-7775/BaseInfo.i.h ROOT-7775/CLIDRegistry.h ROOT-7775/part1.h ROOT-7775/prelude.h \
          ROOT-7775/BaseInfoEmpty.icc ROOT-7775/CaloCell_ID.h ROOT-7775/part2.h

execROOT-7775.log: $(HEADERS)


