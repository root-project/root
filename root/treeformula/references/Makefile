local_clean = Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(DllSuf) *.root $(ALL_LIBRARIES)

CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += refTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Event.mk

# General indication for the size of the files
size=6
tracks=30
# Use compression or not
comp=0
# For writing of the files
action=1

DATAFILES = Event.new.split0.root Event.new.split1.root \
	Event.new.split2.root  Event.new.split9.root 

LOPT:=-g3

refData : $(DATAFILES)

Event.new.split0.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) 0 $(action) $(tracks) >/dev/null && \
	mv Event.root Event.new.split0.root && \
	echo  "$@ made"

Event.new.split1.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) 1 $(action) $(tracks) >/dev/null  && \
	mv Event.root Event.new.split1.root && \
	echo  "$@ made"

Event.new.split2.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) 2 $(action) $(tracks) >/dev/null  && \
	mv Event.root Event.new.split2.root && \
	echo  "$@ made"

Event.new.split9.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) 9 $(action) $(tracks) >/dev/null && \
	mv Event.root Event.new.split9.root && \
	echo  "$@ made"

Event.old.streamed.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) -1 $(action) $(tracks) >/dev/null && \
	mv Event.root Event.old.streamed.root && \
	echo  "$@ made"

Event.old.split.root : Event$(ExeSuf) libEvent.$(DllSuf)
	@./Event $(size) $(comp) -2 $(action) $(tracks) >/dev/null && \
	mv Event.root Event.old.split.root && \
	echo  "$@ made"

refTest: refData ./libEvent.$(DllSuf)
	$(CMDECHO) root.exe -b -l -q dt_RefProxyTest.C > RefProxyTest.log 2>&1 || \
		diff -b RefProxyTest.ref RefProxyTest.log
