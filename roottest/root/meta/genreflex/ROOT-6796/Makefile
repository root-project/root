CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *pcm *o *rflx.* dummy* *rootmap

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

GENREFLEXCXXFLAGS += --rootmap al.rootmap --rootmap-lib libattributesFromTypedef_dictrflx.$(DllSuf)

execattributesFromTypedef.log: libattributesFromTypedef_dictrflx.$(DllSuf)

