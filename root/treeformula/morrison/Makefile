all: tests
test: tests

SUBDIRS =  
TEST_TARGETS += ArraySyntax
CLEAN_TARGETS += *Dict* *.root main$(ExeSuf)

include ../../Rules.mk

#ArraySyntax: Event.root
#	@echo built

ArraySyntax: main$(ExeSuf) $(ROOTSYS)/lib/libTreePlayer.$(DllSuf)
	$(CMDECHO) ./main

main$(ExeSuf): main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) $(CXX) -o$@ $(CXXFLAGS) $^ -I. $(LDSTARTOPT) $(shell eval root-config --libs)

%Dict.C: %.h
	$(CMDECHO) rootcint -f $@ -c $^


