include ../../Rules.mk

tests: dicto

CLEAN_TARGET += 

MyClassDict.C: Header.h Linkdef.h $(ROOTSYS)/bin/rootcint
	rootcint -f MyClassDict.C -c Header.h Linkdef.h

TeleDict.C: Tele.h TeleLinkdef.h $(ROOTSYS)/bin/rootcint
	test ! `rootcint -f TeleDict.C -c Tele.h TeleLinkdef.h` && touch TeleDict.C



dicto : MyClassDict.o TeleDict.o

%.o: %.C
	$(CXX) $(CXXFLAGS) -c $^
