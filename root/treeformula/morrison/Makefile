all: tests
test: tests


TEST_TARGETS += ArraySyntax
CLEAN_TARGETS += *Dict* *.root main$(ExeSuf)


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk


#ArraySyntax: Event.root
#	@echo built

ArraySyntax: main$(ExeSuf) $(ROOTSYS)/lib/libTreePlayer.$(DllSuf)
	$(CMDECHO) ./main

main$(ExeSuf): main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) $(CXX) -o$@ $(CXXFLAGS) $^ -I. $(LDSTARTOPT) $(shell eval root-config --libs)

%Dict.C: %.h
	$(CMDECHO) rootcint -f $@ -c $^


