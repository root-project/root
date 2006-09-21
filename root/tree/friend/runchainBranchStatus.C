{
t1 = new TTree("t1","");
t2 = new TTree("t2","");
int one;
t1->Branch("a",&one,"one/I");
t1->Branch("b",&one,"one/I");
t2->Branch("c",&one,"one/I");
t2->Branch("b",&one,"one/I");
t1->AddFriend(t2);
t1->SetBranchStatus("c",0);
t1->SetBranchStatus("t2.b",0);
}