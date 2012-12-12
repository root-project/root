///
/// Repro the ResetBranch bug
///

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>

#include <iostream>
#include <string>

using namespace std;

/// The value we will store in the ttree
static int gValue = 0;

///
/// Create a single file with a tree with an integer in it
///
void create_file (const string &fname)
{
	cout << "Making ntuple file " << fname << endl;
#ifdef ClingWorkAroundMissingImplicitAuto
	TFile* f = new TFile (fname.c_str(), "RECREATE");
   
	TTree* t = new TTree ("ntup", "Ntuple of one");
#else
	auto f = new TFile (fname.c_str(), "RECREATE");
   
	auto t = new TTree ("ntup", "Ntuple of one");
#endif
	t->Branch("val", &gValue);
   
	for (int i = 0; i < 10; i++)
	{
		gValue = i;
		t->Fill();
	}
   
	f->Write();
	f->Close();
}

void execresetbranch()
{
	/// Create the two files
	create_file ("n1.root");
	create_file ("n2.root");
   
	/// Create the chain that we will be reading in
#ifdef ClingWorkAroundMissingImplicitAuto
	TChain* c = new TChain("ntup");
#else
	auto c = new TChain("ntup");
#endif
	c->AddFile("n1.root");
	c->AddFile("n2.root");
   
	TBranch *b;
	c->SetBranchAddress("val", &gValue, &b);
   
	int ent = c->GetEntries();
	cout << "Total Entries " << ent << endl;
   
	/// Run through most of the index
	int index = 0;
	for (index = 0; index < ent - 50; index++) {
		c->GetEntry(index);
	}
   
	/// Now, reset the address and then re-go
	// b->ResetAddress(); - doesn't seem to do anything
   // Any operation done directly on a branch of a TTree inside a TChain
   // had unpredicable behavior.
	for (index = 0; index < ent; index++) {
		c->GetEntry(index);
		cout << gValue << endl;
	}
   
   gValue = -1;
	c->ResetBranchAddress(b); // use do not work This doesn't either
	for (index = 0; index < ent; index++) {
		c->GetEntry(index);
		cout << gValue << endl;
	}
	
   gValue = -1;
   c->ResetBranchAddresses(); // THis actuall works!!
	for (index = 0; index < ent; index++) {
		c->GetEntry(index);
		cout << gValue << endl;
	}
   
}
