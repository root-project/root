all: tests
test: tests

SUBDIRS =  
TEST_TARGETS += ArraySyntax
CLEAN_TARGETS += *Dict* *.root main$(ExeSuf)

include ../../Rules.mk

#ArraySyntax: Event.root
#	@echo built

ArraySyntax: main$(ExeSuf) $(ROOTSYS)/lib/libTreePlayer.$(DllSuf)
	./main

main$(ExeSuf): main.C fooDict.C barDict.C foo.C bar.C
	$(CXX) -o$@ $(CXXFLAGS) $^ -I. $(LDSTARTOPT) $(shell eval root-config --libs)

%Dict.C: %.h
	rootcint -f $@ -c $^


