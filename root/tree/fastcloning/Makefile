CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog copiedEvent* Event.h one two merged *.root
TEST_TARGETS += single merge
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
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

$(REFFILE) : Event.new.split9.root copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("Event.new.split9.root")' > event.new.split9.root.clone.log 2>&1 && touch $@

#dt_MakeRef.C dt_DrawTest.C libEvent.$(DllSuf) Event.h Event.new.split9.root
#        @root -b -q 'dt_MakeRef.C("Event.new.split9.root",0);' $(OUTPUT) && \
#        echo "$@ made"

export EVENTBUFSIZE = 20
export size = 20

Event.new.split0.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'  > $@.clone.log 2>&1

Event.new.split1.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1

Event.new.split2.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1

Event.new.split9.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1

Event.old.split.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
ifneq ($(FAIL),)
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1
endif

Event.old.streamed.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@ > $@.create.log 2>&1
ifneq ($(FAIL),)
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")' > $@.clone.log 2>&1
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
	$(CMDECHO) cp copiedEvent$(ExeSuf) libEvent.* $*

one/libEvent.$(DllSuf): one/copiedEvent$(ExeSuf)
two/libEvent.$(DllSuf): two/copiedEvent$(ExeSuf)

$(ONEFILES): %: one/copiedEvent$(ExeSuf) one/libEvent.$(DllSuf)
	$(CMDECHO) (cd one; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F)) > $@.create.log 2>&1

$(TWOFILES): %: two/copiedEvent$(ExeSuf) two/libEvent.$(DllSuf)
	$(CMDECHO) (cd two; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F)) > $@.create.log 2>&1

merged/$(REFFILE) : merged/Event.new.split9.root copiedEvent$(ExeSuf) merged/libEvent.$(DllSuf)
	$(CMDECHO) (cd merged; root -l -b -q '../dt_MakeRef.C("Event.new.split9.root",0);' > /dev/null)
#	$(CMDECHO) root.exe -q -b -l 'mtrans.C("Event.new.split9.root")' && touch $@

merged:
	mkdir -p merged

merged/libEvent.$(DllSuf) : copiedEvent$(ExeSuf) libEvent.$(DllSuf)
	$(CMDECHO) mkdir -p merged; cp libEvent.* merged	

$(MERGEFILES): %: $(ONEFILES) $(TWOFILES) 
	$(CMDECHO) mkdir -p merged; root.exe -q -b -l 'mtrans.C("$(@F)")' > $@.create.log 2>&1

merge.log: merged/$(REFFILE) $(MERGEFILES) copiedEvent$(ExeSuf) dt_RunDrawTest_C.$(DllSuf)
	$(CMDECHO) ./dt_RunDrawTest.sh 'merged/' > merge.raw.log 2>&1
	$(CMDECHO) grep -v 'ROOTMARKS' merge.raw.log | grep '^..' > merge.log


testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

single: single.log
	$(TestDiff)

merge: merge.log
	$(TestDiff)
