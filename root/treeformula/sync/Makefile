CLEAN_TARGETS += main *Dict* Event.root *~ Event$(ExeSuf) createfile.out readfile.out libEvent.$(DllSuf) Event.h
TEST_TARGETS += Simple

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


$(ROOTSYS)/test/Event$(ExeSuf):
	(cd $(ROOTSYS)/test; gmake Event$(ExeSuf))

libEvent.$(DllSuf): $(ROOTSYS)/test/libEvent.$(DllSuf)
	$(CMDECHO) cp $(ROOTSYS)/test/Event$(ExeSuf) $(ROOTSYS)/test/libEvent.$(DllSuf) $(ROOTSYS)/test/Event.h .

Event$(ExeSuf): $(ROOTSYS)/test/Event$(ExeSuf)
	$(CMDECHO)cp $(ROOTSYS)/test/Event$(ExeSuf) $(ROOTSYS)/test/libEvent.$(DllSuf) $(ROOTSYS)/test/Event.h .

Event.root: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO) ./Event 20 1 9 1 > createfile.out

Simple: Event.root
ifeq ($(FAIL),)
	$(CMDECHO) echo Simple has a known failure
	$(CMDECHO) root.exe -l -b -q 'Run.C(1)' > readfile.out
else
	$(CMDECHO) root.exe -l -b -q 'Run.C(0)' > readfile.out
endif
