
all: tests
test: tests ChainFriend 

override SUBDIRS =  

CLEAN_TARGETS += main *Dict* *~ createfile.out run.out unevenFriend.out
TEST_TARGETS += ChainFriend unevenFriend

include ../../Rules.mk


files: Event.root Event2.root

unevenFriend: unevenFriend.out
	$(CMDECHO) diff unevenFriend.out unevenFriend.ref

unevenFriend.out:  unevenFriend.C unevenFriend.ref
	$(CMDECHO) root.exe -q -l -b unevenFriend.C > unevenFriend.out

ifeq ($(PLATFORM),win32)
ROOTFILE = $(shell cygpath -u '$(ROOTSYS)/test/Event.root')
else
ROOTFILE = $(ROOTSYS)/test/Event.root
endif

$(ROOTFILE): $(ROOTSYS)/test/Event.h
	$(CMDECHO) ( \
		cd $(ROOTSYS)/test; \
		gmake Event; \
		Event 10 1 9 1; \
	) > createfile.out

Event.root: $(ROOTSYS)/test/Event.root
	$(CMDECHO) cp $(ROOTFILE) .

Event2.root: $(ROOTSYS)/test/Event.root
	$(CMDECHO) cp $(ROOTFILE) Event2.root

ChainFriend: Run.C  $(ROOTSYS)/lib/libTreePlayer.$(DllSuf) files
	$(CMDECHO) root.exe -q -b -l Run.C > run.out 2>&1

main: main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) g++ -g -o $@ $^ -I. `root-config --cflags --libs` -lTreePlayer -lProof

libFoo.so: main.C fooDict.C barDict.C foo.C bar.C
	$(CMDECHO) g++ -g -shared -o $@ $^ -I. `root-config --cflags --libs`

%Dict.C: %.h %Linkdef.h
	$(CMDECHO) rootcint -f $@ -c $^

#clean:
#	rm -f main *Dict* Event.root *~

