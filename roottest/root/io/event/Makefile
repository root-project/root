local_clean = Event$(ExeSuf) libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log *.$(ObjSuf) $(ALL_LIBRARIES) *.root *.lock lock

CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += bigeventTest

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
include $(ROOTTEST_HOME)/scripts/Event.mk

#.SUFFIXES: .cxx .o .so

CXXOPT:=-Z7
LDOPT:=-debug

#all: bigeventTest

bigeventTest.write.log: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO) $(call locked_execution,bigeventTest.write, sleep 10; (./Event 10 1 1 1;) > Event.write.log 2>&1 && echo "write success" > bigeventTest.write.log,test, echo "write success" > bigeventTest.write.log)

bigeventTest.read.log: ./Event$(ExeSuf) ./libEvent.$(DllSuf) bigeventTest.write.log
	$(CMDECHO)./Event 10 1 1 20 > Event.write.log 2>bigeventTest.read.log && echo "read success" > bigeventTest.read.log

bigeventTest.log: bigeventTest.write.log bigeventTest.read.log
	$(CMDECHO) cat bigeventTest.write.log bigeventTest.read.log > bigeventTest.log

bigeventTest.success: bigeventTest.log
	$(CMDECHO) diff -u -b bigeventTest.ref bigeventTest.log && touch bigeventTest.success

bigeventTest: bigeventTest.log bigeventTest.success
	$(TestDiff)

