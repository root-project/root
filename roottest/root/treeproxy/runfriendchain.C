#include "make_chains.cxx"

// Test provided by Scott Fallows (fallows at physics.umn.edu)

void runfriendchain() {
  // test script to illustrate use of MakeProxy with friend trees
  // usage:
  // root[0] .x testvertex.C

  TChain* lchain = make_long_chain();
  TChain* fchain = make_float_chain();
 
  lchain->AddFriend(fchain);

  lchain->MakeProxy("withfriend","vertex.C","","nohist");
  lchain->Process("withfriend.h+","goff",40,480); 
}

