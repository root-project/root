#include <cstdlib>
#include "TTree.h"
#include "TFile.h"
#include "TRandom.h"

#include "Data.h"

int fillTree() {

	Data * pData = new Data();
	TFile f1("myTree.root","Recreate");
	TTree *tr = new TTree("tr", "tr");
	tr->Branch("Data", "Data", &pData);

	for(Int_t i=0; i<100; i++) {

		for(Int_t ins=0; ins<3; ins++) {
         pData->ns[ins].orient = ins;
         for(Int_t isubs=0; isubs<2; isubs++) {
            for(Int_t iefg=0;iefg<5;iefg++) {
               (pData->ns[ins]).subs[isubs].efg[iefg] = gRandom->Rndm() * 20000; // random()/ 10000;
            }
         }
			for(Int_t ich=0; ich<49; ich++) {
				(pData->ns[ins]).adc[ich] = gRandom->Rndm() * 2000000; // random() / 1000 ;
			}
		}

		tr->Fill();
	}

	f1.Write();
	f1.Close();

	return 0;
}
