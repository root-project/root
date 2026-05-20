///
/// Check expect filename formats for TChain::Add and TChain::AddFile work
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
	auto f = new TFile (fname.c_str(), "RECREATE");
	auto t = new TTree ("ntup", "Ntuple of one");
	t->Branch("val", &gValue);
   
	for (int i = 0; i < 10; i++)
	{
		gValue = i;
		t->Fill();
	}
   
	f->Write();
	f->Close();
}

void execfilenameformats()
{
	Long64_t nent;
	TChain *mychain = 0;

	/// Create two files with trees
	create_file ("ff_n1.root");
	create_file ("ff_n2.root");

	cout << "Adding two files to a chain" << endl;
	mychain = new TChain("ntup");
	mychain->AddFile("ff_n1.root");
	mychain->AddFile("ff_n2.root");
	nent = mychain->GetEntries();
	if (nent != 20) {
		cout << "Unexpected number of entries" << endl;
	}

	delete mychain;
	cout << "Adding two files to a chain, with treename specified in the filename" << endl;
	mychain = new TChain("nosuchtree");
	mychain->AddFile("ff_n1.root/ntup");
	mychain->AddFile("ff_n2.root/ntup");
	nent = mychain->GetEntries();
	if (nent != 20) {
		cout << "Unexpected number of entries" << endl;
	}

	delete mychain;
	cout << "Using wildcard to add all maching files to a chain, with treename specified in the filename" << endl;
	mychain = new TChain("nosuchtree");
	mychain->Add("ff_n*.root/ntup");
	nent = mychain->GetEntries();
	if (nent != 20) {
		cout << "Unexpected number of entries" << endl;
	}

	delete mychain;
	mychain = 0;
}
