CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog
TEST_TARGETS += single mytest
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

$(REFFILE) : Event.new.split9.root copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("Event.new.split9.root")' && touch $@

#dt_MakeRef.C dt_DrawTest.C libEvent.$(DllSuf) Event.h Event.new.split9.root
#        @root -b -q 'dt_MakeRef.C("Event.new.split9.root",0);' $(OUTPUT) && \
#        echo "$@ made"

export EVENTBUFSIZE = 20
export size = 200

Event.new.split0.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split1.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split2.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split9.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@

Event.old.split.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.old.streamed.root: copiedEvent
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

single.log:  $(REFFILE) $(DATAFILES) $(ROOTCORELIBS)
	$(CMDECHO) ./dt_RunDrawTest.sh > single.raw.log
	$(CMDECHO) grep -v 'ROOTMARKS' single.raw.log > single.log

FILEDIRS=one two
EVENTDIRS=one/copiedEvent two/copiedEvent

MDATAFILES = Event.new.split0.root Event.new.split1.root \
        Event.new.split2.root  Event.new.split9.root \
        Event.old.streamed.root 
# TChain::Merge does not currently work with old.split.root
# Event.old.split.root

ONEFILES=$(MDATAFILES:%=one/%)
TWOFILES=$(MDATAFILES:%=two/%)
MERGEFILES=$(MDATAFILES:%=merge/%)

$(EVENTDIRS): %/copiedEvent: copiedEvent
	$(CMDECHO) mkdir -p $*
	$(CMDECHO) cp copiedEvent libEvent.$(DllSuf) $*

$(ONEFILES): %: one/copiedEvent one/libEvent.$(DllSuf)
	$(CMDECHO) (cd one; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $(@F))

$(TWOFILES): %: two/copiedEvent two/libEvent.$(DllSuf)
	$(CMDECHO) (cd two; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent $(@F))

merge/$(REFFILE) : merge/Event.new.split9.root copiedEvent merge/libEvent.$(DllSuf)
	$(CMDECHO) (cd merge; root -l -b -q '../dt_MakeRef.C("Event.new.split9.root",0);' > /dev/null)
#	$(CMDECHO) root.exe -q -b -l 'mtrans.C("Event.new.split9.root")' && touch $@

merge:
	mkdir -p merge

merge/libEvent.$(DllSuf) : copiedEvent libEvent.$(DllSuf)
	$(CMDECHO) mkdir -p merge; cp libEvent.$(DllSuf) merge	

$(MERGEFILES): %: $(ONEFILES) $(TWOFILES)
	$(CMDECHO) mkdir -p merge; root.exe -q -b -l 'mtrans.C("$(@F)")'

merged.log: merge/$(REFFILE) $(MERGEFILES) copiedEvent
	$(CMDECHO) ./dt_RunDrawTest.sh 'merge/' > merged.raw.log
	$(CMDECHO) grep -v 'ROOTMARKS' merged.raw.log > merged.log


testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

single: single.log
	$(TestDiff)
