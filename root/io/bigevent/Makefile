.SUFFIXES: .cxx .o .so

all: test

test: ./Event ./libEvent.so

Event : MainEvent.o Event.o EventDict.o
	g++ -o Event MainEvent.o Event.o EventDict.o -L$(ROOTSYS)/lib \
    -lCore -lCint -lTree -lHist -lMatrix -lPhysics -lGraf -lm -ldl 

libEvent.so :  Event.o EventDict.o
	g++ -shared -O Event.o EventDict.o -o  libEvent.so

EventDict.cxx: Event.h EventLinkDef.h $(ROOTSYS)/bin/rootcint $(ROOTSYS)/lib/libCint.so
	rootcint -f EventDict.cxx -c Event.h EventLinkDef.h

Event.o: Event.cxx Event.h

.cxx.o:
	g++ -g -Wall -fPIC -I$(ROOTSYS)/include -c $<

 
