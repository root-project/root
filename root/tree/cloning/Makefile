# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += treeCloneTest.log hsimple1.root hsimple2.root event1.root event2.root

#Set the list of target to make while testing
TEST_TARGETS += treeCloneTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

hsimple1.root: hsimple.C
	$(CMDECHO) root.exe -q -l -b hsimple.C > hsimple.out
	$(CMDECHO) mv hsimple.root hsimple1.root
	$(CMDECHO) cp hsimple1.root hsimple2.root

event1.root: $(ROOTSYS)/test/Event.new.split0.root
	$(CMDECHO) cp $(ROOTSYS)/test/Event.new.split0.root event1.root
	$(CMDECHO) cp event1.root event2.root

treeCloneTest.log: hsimple1.root event1.root
	$(CMDECHO) ( root.exe -q -b -l run.C && root.exe -q -b -l runEvent.C ) > treeCloneTest.log 2>&1

treeCloneTest: treeCloneTest.log treeCloneTest.good 
	$(CMDECHO) diff -ub treeCloneTest.good treeCloneTest.log

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


