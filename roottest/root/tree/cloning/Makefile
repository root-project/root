# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.log hsimple1.root hsimple2.root event1.root event2.root merge.root \
	pion_merged.root merged.root Event.new.split0.root \
        output_Coulomb_LER_study_small.root \
        CopyTree.root Event.h copiedEvent hsimple.out index*.root $(ALL_LIBRARIES) *.clog


#Set the list of target to make while testing
TEST_TARGETS += deepClass mergeWithDir treeCloneTest treeCloneTest2 index

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


Event.new.split0.root: copiedEvent$(ExeSuf) $(ROOTCORELIBS) 
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) Event.new.split0.root

ifneq ($(ClingWorkAroundMissingAutoLoading),)
hsimple1.root: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

hsimple1.root: hsimple.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b hsimple.C > hsimple.out
	$(CMDECHO) mv hsimple.root hsimple1.root
	$(CMDECHO) cp hsimple1.root hsimple2.root

event1.root: Event.new.split0.root 
	$(CMDECHO) cp Event.new.split0.root event1.root
	$(CMDECHO) cp event1.root event2.root

ifneq ($(ClingWorkAroundBrokenUnnamedReturn),)
treeCloneTest.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

treeCloneTest.log: hsimple1.root event1.root runEvent.C run.C  $(ROOTCORELIBS) 
	$(CMDECHO) ( $(CALLROOTEXE) -q -b -l run.C && $(CALLROOTEXE) -q -b -l runEvent.C ) > treeCloneTest.log 2>&1

treeCloneTest: treeCloneTest.log treeCloneTest.ref
	$(CMDECHO) diff -ub treeCloneTest.ref treeCloneTest.log

treeCloneTest2.log: runtreeCloneTest2.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runtreeCloneTest2.C 2>&1 | grep -v Adding > treeCloneTest2.log

treeCloneTest2: treeCloneTest2.log 
	$(TestDiff)

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

mergeWithDir: mergeWithDir.log
	$(TestDiff)

CopyTree.root: make_CopyTree.C ver_40200.root $(ROOTCORELIBS) 
	$(CMDECHO) $(CALLROOTEXE) -q -b -l make_CopyTree.C > make_CopyTree.log 2>&1

deepClass.log: CopyTree.root

ifneq ($(ClingWorkAroundBrokenUnnamedReturn),)
deepClass: CINT_VERSION := brokenReturn
endif

deepClass: deepClass.log
	$(TestDiffCintSpecific)

index: index.clog
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
treeCloneTest2.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif
