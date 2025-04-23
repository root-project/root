{
gROOT->ProcessLine(".x friendOfFriend.C+");
gROOT->ProcessLine(".L testFriends.C+");
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine("testChainFriends();");
#else
testChainFriends();
#endif
}
