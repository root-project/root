# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log hsimple1.root hsimple2.root event1.root event2.root merge.root \
	CopyTree.root Event.h copiedEvent hsimple.out


#Set the list of target to make while testing
TEST_TARGETS += deepClass mergeWithDir treeCloneTest treeCloneTest2

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


Event.new.split0.root: copiedEvent$(ExeSuf) $(ROOTCORELIBS) 
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) Event.new.split0.root

hsimple1.root: hsimple.C
	$(CMDECHO) root.exe -q -l -b hsimple.C > hsimple.out
	$(CMDECHO) mv hsimple.root hsimple1.root
	$(CMDECHO) cp hsimple1.root hsimple2.root

event1.root: Event.new.split0.root 
	$(CMDECHO) cp Event.new.split0.root event1.root
	$(CMDECHO) cp event1.root event2.root

treeCloneTest.log: hsimple1.root event1.root runEvent.C run.C  $(ROOTCORELIBS) 
	$(CMDECHO) ( root.exe -q -b -l run.C && root.exe -q -b -l runEvent.C ) > treeCloneTest.log 2>&1

treeCloneTest: treeCloneTest.log treeCloneTest.good 
	$(CMDECHO) diff -ub treeCloneTest.good treeCloneTest.log

treeCloneTest2.log: runtreeCloneTest2.C $(ROOTCORELIBS)
	$(CMDECHO) root.exe -b -q -l runtreeCloneTest2.C 2>&1 | grep -v Adding > treeCloneTest2.log

treeCloneTest2: treeCloneTest2.log 
	$(TestDiff)

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

mergeWithDir: mergeWithDir.log
	$(TestDiff)

CopyTree.root: make_CopyTree.C ver_40200.root $(ROOTCORELIBS) 
	$(CMDECHO) root.exe -q -b -l make_CopyTree.C > make_CopyTree.log 2>&1

deepClass.log: CopyTree.root

deepClass: deepClass.log
	$(TestDiff)
