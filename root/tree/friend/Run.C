bool TestError(char *test, char *msg) {
  fprintf(stderr,"In test \"%s\", failed: %s\n",
          test,msg);
  return false;
}

bool testTreeByName() {
  const char* tname = "testTreeByName";

  TFile * file = new TFile("Event.root");
  if (!file) return false;
  TTree * tree = (TTree*)file->Get("T");
  if (!tree) return false;

  tree->AddFriend("T2 = T","Event2.root");

  TLeaf* l = tree->GetLeaf("event");
  if (!l) return TestError(tname,"retreving local leaf \"event\" ");

  l = tree->GetLeaf("T2.event");
  if (!l) return TestError(tname,"retreving local leaf \"T2.event\" ");

  delete file;
  return true;
}

bool testTreeByPointer() {
  const char* tname = "testTreeByPointer";

  TFile * file = new TFile("Event.root");
  if (!file) return false;
  TTree * tree = (TTree*)file->Get("T");
  if (!tree) return false;

  TFile * file2 = new TFile("Event2.root");
  if (!file2) return false;
  TTree * tree2 = (TTree*)file2->Get("T");
  if (!tree2) return false;

  tree->AddFriend(tree2,"T2");

  TLeaf* l = tree->GetLeaf("event");
  if (!l) return TestError(tname,"retreving local leaf \"event\" ");

  l = tree->GetLeaf("T2.event");
  if (!l) return TestError(tname,"retreving local leaf \"T2.event\" ");

  delete file;
  return true;
}

bool testChainByName() {
  const char* tname = "testChainByName";

  TChain *c1 = new TChain("T");
  c1->Add("Event.root");
  c1->Add("Eventa.root");
  
  TChain *c2 = new TChain("T2");
  c2->Add("Event2.root/T");
  c2->Add("Event2a.root/T");

  c1->AddFriend("T2");

  TLeaf* l = c1->GetLeaf("event");
  if (!l) return TestError(tname,"retreving local leaf \"event\" ");

  l = c1->GetLeaf("T2.event");
  if (!l) return TestError(tname,"retreving local leaf \"T2.event\" ");

  c1->Draw("T2.event.fNtrack");

  delete c1;
  // do not delete .. it is already done by c1. delete c2;
  return true;
}

bool testChainDifferent() {
  const char* tname = "testChainByName";

  TChain *c1 = new TChain("T");
  c1->Add("Event.root");
  //c1->Add("Eventa.root");
  
  TChain *c2 = new TChain("T2");
  c2->Add("Event3a.root/T");
  c2->Add("Event3b.root/T");

  c1->AddFriend("T2");

  TLeaf* l = c1->GetLeaf("event");
  if (!l) return TestError(tname,"retreving local leaf \"event\" ");

  l = c1->GetLeaf("T2.event");
  if (!l) return TestError(tname,"retreving local leaf \"T2.event\" ");

  c1->Draw("T2.event.fNtrack");

  delete c1;
  // do not delete .. it is already done by c1. delete c2;
  return true;
}

bool testChainByPointer() {
  const char* tname = "testChainByPointer";

  TChain *c1 = new TChain("T");
  c1->Add("Event.root");
  
  TChain *c2 = new TChain("T2");
  c2->Add("Event2.root/T");

  c1->AddFriend(c2);
  c1->AddFriend(c2,"T3");

  TLeaf* l = c1->GetLeaf("event");
  if (!l) return TestError(tname,"retreving local leaf \"event\" ");

  l = c1->GetLeaf("T2.event");
  if (!l) return TestError(tname,"retreving local leaf \"T2.event\" ");

  l = c1->GetLeaf("T3.event");
  if (!l) return TestError(tname,"retreving local leaf \"T3.event\" ");

  delete c1;
  // do not delete .. it is already done by c1. delete c2;
  return true;
}



int Run() {
  bool result = true;
  result &= testTreeByName();
  result &= testTreeByPointer();
  result &= testChainByName();
  result &= testChainByPointer();
  result &= testChainDifferent();

  exit(!result);
}
