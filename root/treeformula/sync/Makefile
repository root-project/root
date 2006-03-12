CLEAN_TARGETS += main *Dict* Event.root *~ Event$(ExeSuf) createfile.out readfile.out libEvent.$(DllSuf) Event.h $(ALL_LIBRARIES) *.manifest
TEST_TARGETS += load Simple

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


$(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf):
	(cd $(ROOTTEST_HOME)/root/io/event; $(MAKE) Event$(ExeSuf))

libEvent.$(DllSuf): $(ROOTTEST_HOME)/root/io/event/libEvent.$(DllSuf)
ifeq ($(PLATFORM),win32)
	$(CMDECHO) if [ -e $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf).manifest ] ; then cp $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf).manifest . ; fi
endif
	$(CMDECHO) cp $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf) $(ROOTTEST_HOME)/root/io/event/libEvent.* $(ROOTTEST_HOME)/root/io/event/Event.h .

Event$(ExeSuf): $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf)
ifeq ($(PLATFORM),win32)
	$(CMDECHO) if [ -e $(EVENTDIR)/Event$(ExeSuf).manifest ] ; then cp $(EVENTDIR)/Event$(ExeSuf).manifest ./copiedEvent$(ExeSuf).manifest ; fi
endif
	$(CMDECHO)cp $(ROOTTEST_HOME)/root/io/event/Event$(ExeSuf)* $(ROOTTEST_HOME)/root/io/event/libEvent.* $(ROOTTEST_HOME)/root/io/event/Event.h .

Event.root: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO) ./Event 20 1 9 1 > createfile.out

Simple: Event.root
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO) root.exe -l -b -q 'Run.C(1)' > readfile.out
else
	$(CMDECHO) root.exe -l -b -q 'Run.C(0)' > readfile.out
endif

load.log: loadcode_C.$(DllSuf)

load: load.log
	$(TestDiff)
