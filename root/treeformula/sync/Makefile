all: tests
test: tests

SUBDIRS =  

TEST_TARGETS = $(SUBDIRS:%=%.test) 

tests: $(TEST_TARGETS) Simple
	@echo "All test succeeded in `pwd`"

$(ROOTSYS)/test/Event:
	(cd $ROOTSYS/test; gmake Event)

libEvent.so: $(ROOTSYS)/test/libEvent.so
	cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event: $(ROOTSYS)/test/Event
	cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event.root: ./Event ./libEvent.so
	./Event 20 1 9 1

Simple: Event.root
	@echo 'gROOT->ProcessLine(".L sync.C");gSystem->Exit(!sync());'  | root.exe -l -b

clean:
	rm -f main *Dict* Event.root *~ Event
