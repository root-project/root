ROOTTEST_COMPILE_MACRO(zmumuSelDrawmail.cpp
                       FIXTURES_SETUP root-tree-friend-zmumuSelDrawmail-fixture)

ROOTTEST_ADD_TEST(treeChainFriend
                  MACRO zmumuSelDrawmail.cpp+
                  COPY_TO_BUILDDIR data12_Apr2001_h1930_on-84-100.root
                                   data12_Feb2000_h1900_off-3-100.root
                  OUTREF treeChainFriend.ref
                  FIXTURES_REQUIRED root-tree-friend-zmumuSelDrawmail-fixture)

ROOTTEST_ADD_TEST(unevenFriend
                  MACRO unevenFriend.C
                  COPY_TO_BUILDDIR short0.root short1.root long.root
                  OUTREF unevenFriend.ref)

ROOTTEST_ADD_TEST(file
                  COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent 10 1 9 1
                  COPY_TO_BUILDDIR MC_uds_reco-1.root MC_uds_reco-2.root
                  FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                  FIXTURES_SETUP root-tree-friend-file-fixture)

ROOTTEST_ADD_TEST(file2
                  COMMAND ${CMAKE_COMMAND} -E copy Event.root Event2.root
                  FIXTURES_REQUIRED root-tree-friend-file-fixture
                  FIXTURES_SETUP root-tree-friend-file2-fixture)

ROOTTEST_ADD_TEST(mcfiles
                  COMMAND ${CMAKE_COMMAND} -E echo "Copy MC files"
                  COPY_TO_BUILDDIR MC_uds_reco-1.root MC_uds_reco-2.root
                  FIXTURES_SETUP root-tree-friend-mcfiles-fixture)

ROOTTEST_ADD_TEST(ChainFriend
                  MACRO runChainFriend.C
                  OUTREF ChainFriend.ref
                  COPY_TO_BUILDDIR Eventa.root
                                   Event2a.root
                                   Event3a.root
                                   Event3b.root
                  FIXTURES_REQUIRED root-tree-friend-file-fixture root-tree-friend-file2-fixture)


ROOTTEST_ADD_TEST(testFriend
                  MACRO testFriend.cc
                  FIXTURES_REQUIRED root-tree-friend-file-fixture
                                    root-tree-friend-mcfiles-fixture)
ROOTTEST_ADD_TEST(testFriend2
                  MACRO testFriend2.cc
                  FIXTURES_REQUIRED root-tree-friend-file-fixture
                                    root-tree-friend-mcfiles-fixture)
ROOTTEST_ADD_TEST(testFriend3
                  MACRO testFriend3.cc
                  FIXTURES_REQUIRED root-tree-friend-file-fixture
                                    root-tree-friend-mcfiles-fixture)
ROOTTEST_ADD_TEST(testFriend4
                  MACRO testFriend4.cc
                  FIXTURES_REQUIRED root-tree-friend-file-fixture
                                    root-tree-friend-mcfiles-fixture)

ROOTTEST_COMPILE_MACRO(friendOfFriend.C
                       FIXTURES_SETUP root-tree-friend-friendOfFriend-fixture)

ROOTTEST_COMPILE_MACRO(testFriends.C
                       FIXTURES_SETUP root-tree-friend-testFriends-fixture)

ROOTTEST_ADD_TEST(FriendOfFriends
                  MACRO runFriendOfFriends.C
                  OUTREF FriendOfFriends.ref
                  FIXTURES_REQUIRED root-tree-friend-friendOfFriend-fixture
                                    root-tree-friend-testFriends-fixture)

ROOTTEST_ADD_TEST(treefriend
                  MACRO runtreefriend.C
                  OUTREF treefriend.ref)

ROOTTEST_COMPILE_MACRO(testFriendsIndices.C
                       FIXTURES_SETUP root-tree-friend-testFriendsIndices-fixture)

ROOTTEST_ADD_TEST(FriendsIndices
                  MACRO testFriendsIndices.C+
                  MACROARG 7
                  OUTREF FriendsIndices.ref
                  FIXTURES_REQUIRED root-tree-friend-testFriendsIndices-fixture)

ROOTTEST_ADD_TEST(friendInChain
                  MACRO runfriendInChain.C
                  COPY_TO_BUILDDIR dat_001.root dat_002.root dat_003.root
                  OUTREF friendInChain.ref)

ROOTTEST_ADD_TEST(chainBranchStatus
                  MACRO runchainBranchStatus.C
                  OUTREF chainBranchStatus.ref)

ROOTTEST_ADD_TEST(circular
                  MACRO runcircular.C
                  OUTREF circular.ref)

ROOTTEST_ADD_TEST(localfriend
                  MACRO runlocalfriend.C
                  OUTREF localfriend.ref)

ROOTTEST_ADD_TEST(UnevenChain
                  MACRO runUnevenChain.C
                  OUTREF UnevenChain.ref)

ROOTTEST_ADD_TEST(Unaligned
                  MACRO runUnaligned.C
                  OUTREF Unaligned.ref)

ROOTTEST_ADD_TEST(MissingParentValue
                  MACRO execMissingParentValue.C
                  OUTREF execMissingParentValue.ref)

ROOTTEST_ADD_TEST(NonExistingFriend
                  MACRO execNonExistingFriend.C
                  OUTCNVCMD sed -e "s: file [A-Za-z/\].*[/\]: file :g"
                  OUTREF execNonExistingFriend.ref)
