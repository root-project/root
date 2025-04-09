CLEAN_TARGETS += copiedEvent* libEvent.* Event.h Event.root Event2.root main *Dict* *~ createfile.out run.out unevenFriend.out \
     treeChainFriend.log $(ALL_LIBRARIES) [a-f][0-4].root treeparent.root treefriend.root *.log localfriend.root \
	  chainTwo*.root chainOne*.root
TEST_TARGETS += friendInChain treefriend ChainFriend unevenFriend treeChainFriend ChainFriendStatus FriendOfFriends \
                FriendsIndices chainBranchStatus localfriend circular UnevenChain Unaligned

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk



files: Event.root Event2.root

treeChainFriend.log: zmumuSelDrawmail_cpp.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l zmumuSelDrawmail.cpp+ </dev/null > treeChainFriend.log 2>&1

treeChainFriend: treeChainFriend.log
	$(CMDECHO) diff -b treeChainFriend.ref treeChainFriend.log 

unevenFriend: unevenFriend.out
	$(CMDECHO) diff -b unevenFriend.ref unevenFriend.out

unevenFriend.out:  unevenFriend.C unevenFriend.ref
	$(CMDECHO) $(CALLROOTEXE) -q -l -b unevenFriend.C > unevenFriend.out 2>&1

ROOTFILE := Event.root

$(ROOTFILE): copiedEvent$(ExeSuf)
	$(CMDECHO) ./copiedEvent 10 1 9 1 2>&1 > createfile.out

Event2.root: $(ROOTFILE)
	$(CMDECHO) cp $(ROOTFILE) Event2.root

ChainFriend.log: $(ROOT_LOC)/lib/libTreePlayer.$(LibSuf) files 

ChainFriend: ChainFriend.log 
	$(TestDiff)


#	$(CMDECHO) $(CALLROOTEXE) -q -b -l Run.C > run.out 2>&1

ChainFriendStatus: testFriend4.cc Event.root
	$(CMDECHO) $(CALLROOTEXE) -q -b -l testFriend4.cc  > ChainFriendStatus.log 2>&1

FriendOfFriends.log: friendOfFriend_C.$(DllSuf) testFriends_C.$(DllSuf)

FriendOfFriends: FriendOfFriends.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
treefriend.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
ChainFriend.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

treefriend: treefriend.log
	$(TestDiff)

FriendsIndices.log: testFriendsIndices_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l 'testFriendsIndices.C+(7)' > FriendsIndices.log

FriendsIndices: FriendsIndices.log
	$(TestDiff)

friendInChain: friendInChain.log
	$(TestDiff)

chainBranchStatus: chainBranchStatus.log
	$(TestDiff)

circular: circular.clog
	$(TestDiffW)

localfriend: localfriend.log
	$(TestDiff)

UnevenChain: UnevenChain.log
	$(TestDiff)

Unaligned: Unaligned.log
	$(TestDiff)

execMissingParentValue.success: EXTRA_DIFFOPTS += -w

execNonExistingFriend.log: execNonExistingFriend.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b execNonExistingFriend.C 2>&1 | sed -e 's?: file [A-Za-z/\].*[/\]?: file ?'  > execNonExistingFriend.log || handleError.sh --cmd='Execution of execNonExistingFriend.C > execNonExistingFriend.log' --result=$? --log=execNonExistingFriend.log --test=execNonExistingFriend
