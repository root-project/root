include ../../Rules.mk

all: tests
test: tests ChainFriend

override SUBDIRS =  

#TEST_TARGETS = $(SUBDIRS:%=%.test) 

TEST_TARGETS += ChainFriend

#$(TEST_TARGETS): %.test:
#	@(cd $*; gmake test)

#ArraySyntax: Event.root
#	@echo built

files: Event.root Event2.root

Event.root: $(ROOTSYS)/test/Event.root
	cp `cygpath -u '$(ROOTSYS)/test/Event.root' ` .

Event2.root: $(ROOTSYS)/test/Event.root
	cp `cygpath -u '$(ROOTSYS)/test/Event.root' ` Event2.root

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
