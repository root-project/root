
all: tests
test: tests ChainFriend ChainFriendStatus



CLEAN_TARGETS += main *Dict* *~ createfile.out run.out unevenFriend.out treeChainFriend.log $(ALL_LIBRARIES)
TEST_TARGETS += ChainFriend unevenFriend treeChainFriend ChainFriendStatus

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

override SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk



files: Event.root Event2.root

treeChainFriend.log: zmumuSelDrawmail_cpp.$(DllSuf)
	$(CMDECHO) root.exe -b -q -l zmumuSelDrawmail.cpp+ </dev/null > treeChainFriend.log 2>&1

treeChainFriend: treeChainFriend.log
	$(CMDECHO) diff -b treeChainFriend.ref treeChainFriend.log 

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

