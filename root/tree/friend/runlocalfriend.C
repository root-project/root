{ 
   TFile *f = new TFile("localfriend.root","RECREATE");
   TTree *t = new TTree("main","main");
   TTree *t2 = new TTree("friend","friend");
   t->AddFriend(t2);
   t->GetListOfFriends()->ls();
   f->Write();
   return 0;
}
