{
bool result = true;

TChain *chain = new TChain("T");
chain->Add("event1.root");
chain->Add("event2.root");

// f = new TFile("output.root","RECREATE");
TTree *clone1 = chain->CopyTree("");

TBranch *br = clone1->GetBranch("event");

chain->GetEntry(0);
TTree *clone2 = chain->CopyTree("");

chain->LoadTree(0);
int n = chain->GetTree()->GetEntries();

TTree *clone3 = chain->CloneTree(0);

void *chainadd = chain->GetBranch("event")->GetAddress();
void *clone3add = clone3->GetBranch("event")->GetAddress();

if (chainadd != clone3add) {
   cerr << "clone3 is not connected to the chain\n";
   result = false;
}
chain->LoadTree(n+1);

void *chainadd = chain->GetBranch("event")->GetAddress();
void *clone3add = clone3->GetBranch("event")->GetAddress();
if (chainadd != clone3add) {
   cerr << "clone3 is not well connected to the chain\n";
   result = false;
}

// f->Write();
delete clone3;

TTree *clone4 = chain->CloneTree();
if (!result) gApplication->Terminate(1);

}
