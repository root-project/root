CLEAN_TARGETS += *.log main *Dict* Event.root *~ copiedEvent$(ExeSuf) \
	createfile.out readfile.out libEvent.$(DllSuf)\
	Event.cxx Event.h EventLinkDef.h $(ALL_LIBRARIES) *.manifest \
        jetclass.root example.root 
TEST_TARGETS += load Simple alias

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
	$(CMDECHO) ./copiedEvent 20 1 9 1 > createfile.out

ifneq ($(ClingWorkAroundJITfullSymbolResolution),)
Simple: CALLROOTEXE += -e 'gSystem->Load("libEvent");'
endif
ifneq ($(ClingWorkAroundMissingAutoLoading),)
Simple: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

Simple: Event.root
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO) $(CALLROOTEXE) -l -b -q 'Run.C(1)' > readfile.out 2>&1 && diff -w readfile-1.ref readfile.out
else
	$(CMDECHO) $(CALLROOTEXE) -l -b -q 'Run.C(0)' > readfile.out 2>&1 && diff -w readfile-0.ref readfile.out
endif

load.log: loadcode_C.$(DllSuf)

load: load.log
	$(TestDiff)

alias: alias.clog
	$(TestDiff)

execScanFail.log: CUTpad1pad2.C  CUTpad1tof.C 
