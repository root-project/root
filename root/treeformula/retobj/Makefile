CLEAN_TARGETS += $(local_clean)
TEST_TARGETS += retobjTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


.SUFFIXES: .cxx .o .so


all: rejobjTest

retobjTest: Event.new.split0.root
	@root.exe -q -l -b run.C >| result && diff result result.good

Event.new.split0.root: ./Event ./libEvent.$(DllSuf)
	@echo creating Event.new.split0.root
	@./Event 10 1 0 1 >/dev/null 2>&1  && cp Event.root Event.new.split0.root 

Event : MainEvent.o Event.o EventDict.o
	$(CXX) $(OutPutOpt)Event MainEvent.o Event.o EventDict.o -L$(ROOTSYS)/lib \
    -lCore -lCint -lTree -lHist -lMatrix -lPhysics -lGraf -lm -ldl 

libEvent.$(DllSuf) :  Event.o EventDict.o
	$(CXX) $(SOFLAGS) Event.o EventDict.o $(OutPutOpt)libEvent.$(DllSuf)

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.$(DllSuf)
	rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

.cxx.o:
	$(CXX) $(CXXFLAGS) -I$(ROOTSYS)/include -c $<

local_clean = Event libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~ *.root result


