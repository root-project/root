CLEAN_TARGETS += *.log main *Dict* Event.root *~ copiedEvent$(ExeSuf) \
	createfile.out readfile.out libEvent.$(DllSuf)\
	Event.cxx Event.h EventLinkDef.h $(ALL_LIBRARIES) *.manifest \
        jetclass.root
TEST_TARGETS += load Simple alias

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

Event.root: copiedEvent$(ExeSuf) 
	$(CMDECHO) ./copiedEvent 20 1 9 1 > createfile.out

Simple: Event.root
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO) $(CALLROOTEXE) -l -b -q 'Run.C(1)' > readfile.out
else
	$(CMDECHO) $(CALLROOTEXE) -l -b -q 'Run.C(0)' > readfile.out
endif

load.log: loadcode_C.$(DllSuf)

load: load.log
	$(TestDiff)

alias: alias.clog
	$(TestDiff)
