CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog complexOfileROOT6.root *pcm classWithComplex.root classWithComplex.xml

#TEST_TARGETS +=

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

execwritecomplex.log: execwritecomplex.C
execreadcomplex.log: execwritecomplex.log
execwriteClassWithComplex.log: libclassWithComplex_dictrflx.$(DllSuf)
execreadClassWithComplex.log: execwriteClassWithComplex.log
