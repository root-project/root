CLEAN_TARGETS += main *Dict* Event.root *~ Event createfile.out readfile.out
TEST_TARGETS += Simple

include ../../Rules.mk

$(ROOTSYS)/test/Event:
	(cd $(ROOTSYS)/test; gmake Event)

libEvent.so: $(ROOTSYS)/test/libEvent.so
	$(CMDECHO) cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event: $(ROOTSYS)/test/Event
	$(CMDECHO)cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event.root: ./Event ./libEvent.so
	$(CMDECHO) ./Event 20 1 9 1 > createfile.out

Simple: Event.root
ifeq ($(FAIL),)
	$(CMDECHO) echo Simple has a known failure
	$(CMDECHO) root.exe -l -b -q 'Run.C(1)' > readfile.out
else
	$(CMDECHO) root.exe -l -b -q 'Run.C(0)' > readfile.out
endif
