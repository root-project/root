CLEAN_TARGETS += local_clean
TEST_TARGESTS += bigeventTest

include ../../Rules.mk

.SUFFIXES: .cxx .o .so


all: bigeventTest

bigeventTest: ./Event ./libEvent.$(DllSuf)
	./Event 10 1 1 1
	./Event 10 1 1 20

Event : MainEvent.o Event.o EventDict.o
	$(CXX) $(OutPutOpt)Event MainEvent.o Event.o EventDict.o -L$(ROOTSYS)/lib \
    -lCore -lCint -lTree -lHist -lMatrix -lPhysics -lGraf -lm -ldl 

libEvent.$(DllSuf) :  Event.o EventDict.o
	$(CXX) $(SOFLAGS) Event.o EventDict.o $(OutPutOpt)libEvent.$(DllSuf)

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.$(DllSuf)
	rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

Event.o: Event.cxx Event.h

.cxx.o:
	$(CXX) $(CXXFLAGS) -I$(ROOTSYS)/include -c $<

local_clean:
	rm -f Event libEvent.$(DllSuf) *.o EventDict.cxx EventDict.h *~


