{ 
   TFile *f = new TFile("localfriend.root","RECREATE");
   TTree *t = new TTree("main","main");
   TTree *t2 = new TTree("friend","friend");
   t->AddFriend(t2);
   t->GetListOfFriends()->ls();
   f->Write();
#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}
