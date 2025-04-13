{
auto t1 = new TTree("t1","");
auto t2 = new TTree("t2","");
int one;
t1->Branch("a",&one,"one/I");
t1->Branch("b",&one,"one/I");
t2->Branch("c",&one,"one/I");
t2->Branch("b",&one,"one/I");
t1->AddFriend(t2);
t1->SetBranchStatus("c",0);
t1->SetBranchStatus("t2.b",0);

auto t3 = new TTree("t3","");
t3->Branch("a",&one,"one/I");
t3->Branch("b",&one,"one/I");
auto c1 = new TChain("t1");
c1->AddFriend(t2);
c1->SetBranchStatus("c",0);
c1->SetBranchStatus("t2.b",0);

}