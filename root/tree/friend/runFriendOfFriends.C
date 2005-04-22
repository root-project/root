{
gROOT->ProcessLine(".x friendOfFriend.C+");
gROOT->ProcessLine(".L testFriends.C+");
testChainFriends();
}