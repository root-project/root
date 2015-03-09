CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *pcm *o *rflx.* dummy*

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

exectsenums.log: libtsenums_dictrflx.$(DllSuf)
exectsinclude.log: exectsinclude.C
exectsautoparse.log: exectsautoparse.C

