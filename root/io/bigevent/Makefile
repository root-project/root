CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += bigeventTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


#.SUFFIXES: .cxx .o .so

LOPT:=-g3

#all: bigeventTest

bigeventTest: ./Event ./libEvent.$(DllSuf)
	$(CMDECHO)./Event 10 1 1 1 > Event.write.log
	$(CMDECHO)./Event 10 1 1 20 > Event.read.log

Event : MainEvent.o Event.o EventDict.o
	$(CMDECHO) $(CXX) $(OPT) $(OutPutOpt)Event MainEvent.o Event.o EventDict.o -L$(ROOTSYS)/lib \
    -lCore -lCint -lTree -lHist -lMatrix -lPhysics -lGraf -lm -ldl 

libEvent.$(DllSuf) :  Event.o EventDict.o
	$(CMDECHO) $(CXX) $(OPT) $(SOFLAGS) Event.o EventDict.o $(OutPutOpt)libEvent.$(DllSuf) -L$(ROOTSYS)/lib -lPhysics

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.$(DllSuf)
	$(CMDECHO) rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

local_clean = Event libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.log


