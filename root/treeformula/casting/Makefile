all: tests
test: tests

SUBDIRS =  

TEST_TARGETS = $(SUBDIRS:%=%.test) 

tests: $(TEST_TARGETS) Simple
	@echo "All test succeeded in `pwd`"

$(ROOTSYS)/test/Event:
	(cd $ROOTSYS/test; gmake Event)

Event: $(ROOTSYS)/test/Event
	cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event.root:
	./Event 20 1 9 1

Simple_cxx.so : Simple.h Simple.cxx
	@echo 'gROOT->ProcessLine(".L Simple.cxx++");' | root.exe -l -b 

Simple: Simple_cxx.so
	@echo 'gROOT->ProcessLine(".L Run.C");gSystem->Exit(!Run());'  | root.exe -l -b


clean:
	rm -f main *Dict* Event.root *~ *.so
