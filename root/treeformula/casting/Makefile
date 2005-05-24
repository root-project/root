TEST_TARGETS = Simple 
CLEAN_TARGETS += main *Dict* Event.root *~ $(ALL_LIBRARIES) *.log


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk

$(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf):
	(cd $(ROOTTEST_HOME)/root/io/event; $(MAKE) Event$(ExeSuf))

Event$(ExeSuf): $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf)
	$(CMDECHO) cp $(ROOTSYS)/test/Event$(ExeSuf) $(ROOTSYS)/test/libEvent.$(DllSuf) $(ROOTSYS)/test/Event.h .

Event.root: Event$(ExeSuf)
	$(CMDECHO) ./Event 20 1 9 1

Simple.log: Simple_cxx.$(DllSuf)
	$(CMDECHO) root.exe -l -b -q Run.C > Simple.log

Simple: Simple.log
	$(CMDECHO) diff -ubw Simple.log Simple.ref

