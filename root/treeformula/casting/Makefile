SUBDIRS =  
TEST_TARGETS = Simple 
CLEAN_TARGETS += main *Dict* Event.root *~ *.$(DllSuf) *.exp *.pdb *.d

include ../../Rules.mk

$(ROOTSYS)/test/Event:
	(cd $ROOTSYS/test; gmake Event)

Event: $(ROOTSYS)/test/Event
	cp $(ROOTSYS)/test/Event $(ROOTSYS)/test/libEvent.so $(ROOTSYS)/test/Event.h .

Event.root:
	./Event 20 1 9 1

Simple_cxx.$(DllSuf) : Simple.h Simple.cxx
	$(CMDECHO) root.exe -l -b buildSimple.C 

Simple: Simple_cxx.$(DllSuf)
	$(CMDECHO) root.exe -l -b -q Run.C


