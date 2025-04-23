CLEAN_TARGETS += *.log *.so *.d *pcm *rflx* dummy*.C
TEST_TARGETS += execFindBackPCMNames

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

GENREFLEXCXXFLAGS += -m pcm1.pcm -m pcm2.pcm -m pcm3.pcm -m pcm4.pcm -m pcm5.pcm

execFindBackPCMNames.log: libclassesFindBackPCMNames_dictrflx.$(DllSuf)
