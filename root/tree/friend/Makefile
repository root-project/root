
all: tests
test: tests ChainFriend ChainFriendStatus

override SUBDIRS =  

CLEAN_TARGETS += main *Dict* *~ createfile.out run.out unevenFriend.out
TEST_TARGETS += ChainFriend unevenFriend

include ../../Rules.mk


files: Event.root Event2.root

unevenFriend: unevenFriend.out
	$(CMDECHO) diff -b unevenFriend.out unevenFriend.ref

unevenFriend.out:  unevenFriend.C unevenFriend.ref
	$(CMDECHO) root.exe -q -l -b unevenFriend.C > unevenFriend.out

ifeq ($(shell root-config --arch),win32)
ROOTFILE = $(shell cygpath -u '$(ROOTSYS)/test/Event.root')
ROOTDIR = $(shell cygpath -u '$(ROOTSYS)')
else
ROOTFILE = $(ROOTSYS)/test/Event.root
ROOTDIR = $(ROOTSYS)
endif

$(ROOTFILE): $(ROOTSYS)/test/Event.h
	$(CMDECHO) ( \
		cd $(ROOTDIR)/test; \
		gmake Event$(ExeSuf); \
		./Event 10 1 9 1; \
	) > createfile.out

Event.root: $(ROOTDIR)/test/Event.root
	$(CMDECHO) cp $(ROOTFILE) .

Event2.root: $(ROOTDIR)/test/Event.root
	$(CMDECHO) cp $(ROOTFILE) Event2.root

ChainFriend: Run.C  $(ROOTSYS)/lib/libTreePlayer.$(DllSuf) files
	$(CMDECHO) root.exe -q -b -l Run.C > run.out 2>&1

ChainFriendStatus: testFriend4.cc
	$(CMDECHO) root.exe -q -b -l testFriend4.cc > ChainFriendStatus.log 

main: main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) g++ -g -o $@ $^ -I. `root-config --cflags --libs` -lTreePlayer -lProof

libFoo.so: main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) g++ -g -shared -o $@ $^ -I. `root-config --cflags --libs`

%Dict.C: %.h %Linkdef.h
	$(CMDECHO) rootcint -f $@ -c $^

#clean:
#	rm -f main *Dict* Event.root *~

