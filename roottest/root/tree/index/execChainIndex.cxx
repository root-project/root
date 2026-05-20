#include "TChain.h"
#include <vector>
#include <iostream>
using namespace std;
void execChainIndex(){

	TChain *chain = new TChain("SEL_TRACKS","SEL_TRACKS");
	TChain *Bschain = new TChain("Bs","Bs");
	vector<TString> filelist;


 if (0) {
	filelist.push_back("user.abarton.003763.EXT0._00001.ntuple.root");
	filelist.push_back("user.abarton.003763.EXT0._00003.ntuple.root.1");
	filelist.push_back("user.abarton.003763.EXT0._00006.ntuple.root");
 } else {
	filelist.push_back("abarton.1.root");
        filelist.push_back("abarton.2.root");
        filelist.push_back("abarton.3.root");
 }
        for(unsigned int i =0;i<filelist.size();i++){
		chain->Add(filelist[i]);
		Bschain->Add(filelist[i]);
	}	
	
	

	chain->BuildIndex("SEL_TRACKS_runNumber", "SEL_TRACKS_eventNumber");
	const int entries = Bschain->GetEntries();
	int run, event;
	int run2, event2;
	Bschain->SetBranchAddress("Bs_runNumber",&run);
	Bschain->SetBranchAddress("Bs_eventNumber",&event);
	

	chain->SetBranchAddress("SEL_TRACKS_runNumber",&run2);
	chain->SetBranchAddress("SEL_TRACKS_eventNumber",&event2);	

	for(int i =0; i < entries;i++){
		Bschain->GetEntry(i);
		int ind = chain->GetEntryNumberWithIndex(run,event);
		chain->GetEntry(ind);
		cout << "looked up " << run << " " << event << " got " << ind;
		if(event != event2 || run != run2){
			cout << " - WRONG" << endl;
		}else cout << "- correct "<< endl;
	}

}
