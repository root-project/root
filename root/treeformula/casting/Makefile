TEST_TARGETS = Simple 
CLEAN_TARGETS += main *Dict* Event.root *~ $(ALL_LIBRARIES) *.log


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk

$(ROOTSYS)/test/Event:
	(cd $ROOTSYS/test; gmake Event)

Event: $(ROOTSYS)/test/Event
	$(CMDECHO) cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event.root:
	$(CMDECHO) ./Event 20 1 9 1

Simple.log: Simple_cxx.$(DllSuf)
	$(CMDECHO) root.exe -l -b -q Run.C > Simple.log

Simple: Simple.log
	$(CMDECHO) diff -u Simple.log Simple.ref

