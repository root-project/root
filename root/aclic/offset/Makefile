include ../../Rules.mk

all: tests
test: tests TOffset_cxx.$(DllSuf)

override SUBDIRS =  

#TEST_TARGETS = $(SUBDIRS:%=%.test) 

TEST_TARGETS += ChainFriend

#$(TEST_TARGETS): %.test:
#	@(cd $*; gmake test)

#ArraySyntax: Event.root
#	@echo built

TOffset_cxx.$(DllSuf) : TOffset.cxx TOffset.h $(ROOTSYS)/lib/libCore.$(DllSuf)
	root -q -b -l TOffset.cxx++

ChainFriend: Run.C  $(ROOTSYS)/lib/libTreePlayer.$(DllSuf) files
	root.exe -q -b -l Run.C

main: main.C fooDict.C barDict.C foo.C bar.C
	g++ -g -o $@ $^ -I. `root-config --cflags --libs` -lTreePlayer -lProof

libFoo.so: main.C fooDict.C barDict.C foo.C bar.C
	g++ -g -shared -o $@ $^ -I. `root-config --cflags --libs`

%Dict.C: %.h %Linkdef.h
	rootcint -f $@ -c $^

#clean:
#	rm -f main *Dict* Event.root *~

CLEAN_TARGET += main *Dict* Event.root *~
