#Disable until boost is not present in the headers
CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += execLibLoad

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

GENREFLEXCXXFLAGS += --interpreteronly -I ./ -I CoralBase
CXXFLAGS += -I ./ -I CoralBase 

execLibLoad.log: PyCool_rflx.$(ObjSuf)
