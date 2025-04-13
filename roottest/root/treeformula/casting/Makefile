TEST_TARGETS = Simple 
CLEAN_TARGETS += main *Dict* Event.root *~ $(ALL_LIBRARIES) *.log simple.root


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

Event.root: copiedEvent$(ExeSuf)
	$(CMDECHO) ./copiedEvent 20 1 9 1

Simple.log: Simple_cxx.$(DllSuf) Read.C Create.C Run.C
ifneq ($(ClingWorkAroundCallfuncAndInline),)
	$(CMDECHO) $(CALLROOTEXE) -l -b -q Run.C 2>&1 | grep -v GetVisibility > Simple.log 
else
	$(CMDECHO) $(CALLROOTEXE) -l -b -q Run.C > Simple.log 2>&1
endif

Simple: Simple.log
	$(TestDiffCintSpecific)


