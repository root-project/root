CXX = g++
CXXFLAGS = $(shell root-config --cflags)
LDFLAGS = $(shell root-config --libs)


RtObjDict.C: RtObj.h RtObjLinkDef.h $(ROOTSYS)/bin/rootcint
	rootcint -f RtObjDict.C -c RtObj.h RtObjLinkDef.h

.C.o:
	$(CXX) $(CXXFLAGS) -c $<

tester: tester.o RtObjDict.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o tester tester.o RtObjDict.o

