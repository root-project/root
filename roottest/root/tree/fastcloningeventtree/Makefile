CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog copiedEvent* Event.h one two \
   merged/Event* merged/libEvent* merged/dt_reference.root \
   Event*.root dt_reference.root CloneTree.root cal.root Tuple_merge.root TestFile2.root \
   output.root TestMerged.root mix_merge.root mix_int.root mix_char.root execmisstop.root \
   TestFile1.root Tuple_merge.root missplit-1.root missplit-99.root missplit.fail.root


TEST_TARGETS += cloneChain single merge Plot SplitMismatch outoforder filemergererror badmix
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

# Outputs a message if the FAIL variable is null
testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

DATAFILES = Event.new.split0.root Event.new.split1.root \
        Event.new.split2.root  Event.new.split9.root \
        Event.old.streamed.root Event.old.split.root

REFFILE=dt_reference.root
export EVENTBUFSIZE = 20
export size = 20

$(REFFILE) : Event.new.split9.root copiedEvent$(ExeSuf)
	$(CMDECHO) touch Event.new.split9.root # to prevent any potential re-build by -f $(EVENTDIR)/dt_Makefile
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@ > $@.create.log 2>&1
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("Event.new.split9.root")' > event.new.split9.root.clone.log 2>&1 && touch $@

#dt_MakeRef.C dt_DrawTest.C libEvent.$(DllSuf) Event.h Event.new.split9.root
#        @$(CALLROOTEXE) -l -b -q 'dt_MakeRef.C("Event.new.split9.root",0);' $(OUTPUT) && \
#        echo "$@ made"

Event.new.split0.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("$@")'  > $@.clone.log 2>&1

Event.new.split1.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1

Event.new.split2.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1

Event.new.split9.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1

Event.old.split.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
ifneq ($(FAIL),)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1
endif

Event.old.streamed.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
ifneq ($(FAIL),)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1
endif

single.log:  $(REFFILE) $(DATAFILES) $(ROOTCORELIBS) dt_RunDrawTest_C.$(DllSuf)
	$(CMDECHO) ./dt_RunDrawTest.sh  > single.raw.log 2>&1 
	$(CMDECHO) grep -v 'ROOTMARKS' single.raw.log  | grep '^..' > single.log

FILEDIRS=one two
EVENTDIRS=one/copiedEvent$(ExeSuf) two/copiedEvent$(ExeSuf)

ifneq ($(FAIL),)
MDATAFILES = Event.new.split0.root Event.new.split1.root \
        Event.new.split2.root  Event.new.split9.root \
        Event.old.streamed.root Event.old.split.root
else
MDATAFILES = Event.new.split0.root Event.new.split1.root \
        Event.new.split2.root  Event.new.split9.root \
        Event.old.streamed.root 
endif
# TChain::Merge does not currently work with old.split.root
# Event.old.split.root

ONEFILES=$(MDATAFILES:%=one/%)
TWOFILES=$(MDATAFILES:%=two/%)
MERGEFILES=$(MDATAFILES:%=merged/%)

$(EVENTDIRS): %/copiedEvent$(ExeSuf): copiedEvent$(ExeSuf)
	$(CMDECHO) mkdir -p $*
	$(CMDECHO) cp copiedEvent$(ExeSuf)* libEvent.* $*

one/libEvent.$(DllSuf): one/copiedEvent$(ExeSuf)
two/libEvent.$(DllSuf): two/copiedEvent$(ExeSuf)

$(ONEFILES): %: one/copiedEvent$(ExeSuf) one/libEvent.$(DllSuf)
	$(CMDECHO) (cd one; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F)) > $@.create.log 2>&1

$(TWOFILES): %: two/copiedEvent$(ExeSuf) two/libEvent.$(DllSuf)
	$(CMDECHO) (cd two; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F)) > $@.create.log 2>&1

merged/$(REFFILE) : merged/Event.new.split9.root copiedEvent$(ExeSuf) merged/libEvent.$(DllSuf)
	$(CMDECHO) (cd merged; $(CALLROOTEXE) -l -b -q '../dt_MakeRef.C("Event.new.split9.root",0);' > /dev/null)
#	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'mtrans.C("Event.new.split9.root")' && touch $@

merged:
	$(CMDECHO) mkdir -p merged ; cp .rootrc merged

merged/libEvent.$(DllSuf) : merged copiedEvent$(ExeSuf)
	$(CMDECHO) cp libEvent.* merged	

$(MERGEFILES): %: $(ONEFILES) $(TWOFILES) | merged
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'mtrans.C("$(@F)")' > $@.create.log 2>&1

merge.log: merged/$(REFFILE) $(MERGEFILES) copiedEvent$(ExeSuf) dt_RunDrawTest_C.$(DllSuf)
	$(CMDECHO) ./dt_RunDrawTest.sh 'merged/' > merge.raw.log 2>&1
	$(CMDECHO) grep -v 'ROOTMARKS' merge.raw.log | grep '^..' > merge.log


testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingDynamicScope),)
single:

merge:
else
single: single.log
	$(TestDiff)

merge: merge.log
	$(TestDiff)

endif

CloneTree.root: make_CloneTree.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l make_CloneTree.C > make_CloneTree.log 2>&1

cloneChain.log: CloneTree.root

cloneChain: cloneChain.log
	$(TestDiff)

Plot.log: recon-v1r030603p6_700000811_00000-00984_calTuple.root

Plot: Plot.log
	$(TestDiff)

SplitMismatch.clog: runSplitMismatch_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runSplitMismatch.C+ 2>&1 | sed -e 's:file .*roottest:file .../roottest:' > SplitMismatch.clog

SplitMismatch: SplitMismatch.clog
	$(TestDiff)

Tuple_merge.root: Tuple_1.root Tuple_2.root $(ROOTCORELIBS)
	$(CMDECHO) hadd -f Tuple_merge.root Tuple_1.root Tuple_2.root > Tuple_merge.log

outoforder.log: Tuple_merge.root

outoforder: outoforder.log	
	$(TestDiff)

filemergererror: filemergererror.log	
	$(TestDiff)

badmix: badmix.log
	$(TestDiff)


abstract.root: abstract_C.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q 'abstract.C+(0)' > abstract_root.log 2>&1

copy-abstract.root: abstract_C.$(DllSuf) abstract.root
	$(CMDECHO) root.exe -b -l -q 'abstract.C+(1)' > abstract_root.log 2>&1

abstract.log: copy-abstract.root

abstract: abstract.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
filemergererror.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif
