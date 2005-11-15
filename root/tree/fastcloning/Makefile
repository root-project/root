CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog copiedEvent* one two merge 
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
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("Event.new.split9.root")' && touch $@

#dt_MakeRef.C dt_DrawTest.C libEvent.$(DllSuf) Event.h Event.new.split9.root
#        @root -b -q 'dt_MakeRef.C("Event.new.split9.root",0);' $(OUTPUT) && \
#        echo "$@ made"

export EVENTBUFSIZE = 20
export size = 200

Event.new.split0.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split1.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split2.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'

Event.new.split9.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@

Event.old.split.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
ifneq ($(FAIL),)
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'
endif

Event.old.streamed.root: copiedEvent$(ExeSuf)
	$(CMDECHO) $(MAKE) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $@
ifneq ($(FAIL),)
	$(CMDECHO) root.exe -q -b -l 'ctrans.C("$@")'
endif

single.log:  $(REFFILE) $(DATAFILES) $(ROOTCORELIBS)
	$(CMDECHO) ./dt_RunDrawTest.sh > single.raw.log
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
	$(CMDECHO) cp copiedEvent$(ExeSuf) libEvent.$(DllSuf) $*

$(ONEFILES): %: one/copiedEvent$(ExeSuf) one/libEvent.$(DllSuf)
	$(CMDECHO) (cd one; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F))

$(TWOFILES): %: two/copiedEvent$(ExeSuf) two/libEvent.$(DllSuf)
	$(CMDECHO) (cd two; $(MAKE) size=$(size) --no-print-directory -f $(EVENTDIR)/dt_Makefile EVENT=copiedEvent$(ExeSuf) $(@F))

merged/$(REFFILE) : merged/Event.new.split9.root copiedEvent$(ExeSuf) merged/libEvent.$(DllSuf)
	$(CMDECHO) (cd merged; root -l -b -q '../dt_MakeRef.C("Event.new.split9.root",0);' > /dev/null)
#	$(CMDECHO) root.exe -q -b -l 'mtrans.C("Event.new.split9.root")' && touch $@

merged:
	mkdir -p merged

merged/libEvent.$(DllSuf) : copiedEvent$(ExeSuf) libEvent.$(DllSuf)
	$(CMDECHO) mkdir -p merged; cp libEvent.$(DllSuf) merged	

$(MERGEFILES): %: $(ONEFILES) $(TWOFILES) 
	$(CMDECHO) mkdir -p merged; root.exe -q -b -l 'mtrans.C("$(@F)")'

merged.log: merged/$(REFFILE) $(MERGEFILES) copiedEvent$(ExeSuf)
	$(CMDECHO) ./dt_RunDrawTest.sh 'merged/' > merged.raw.log
	$(CMDECHO) grep -v 'ROOTMARKS' merged.raw.log | grep '^..' > merged.log


testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

single: single.log
	$(TestDiff)

merge: merged.log
	$(TestDiff)
