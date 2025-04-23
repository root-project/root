
#include <TTree.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TLeafI.h>
#include <TBranch.h>

#include <TRandom.h>

//#include <iostream>
//using std::cout;
//using std::endl;

typedef struct {
	Int_t		njets;
	Float_t		et[20];
	Float_t		pt[20];
} JETS_t;

void runarraysize() {
	
	JETS_t JETS1, JETS2;
	
	JETS1.njets = 0;
	JETS2.njets = 0;
   
	
//ß	TFile *f = new TFile("file.root", "recreate");
	TTree *t = new TTree("finalnt","A FinalNt tree");
	
	TBranch *br1 = t->Branch("JETS1", &JETS1, "njets/I:et[njets]/F:pt[njets]/F" );
   
	br1->GetLeaf("njets")->SetAddress(&JETS1.njets);
	((TLeafI*) br1->GetLeaf("njets"))->SetMaximum(20);
	//br1->GetLeaf("et")->SetLeafCount(br1->GetLeaf("njets"));		//<--
	br1->GetLeaf("et")->SetAddress(JETS1.et);
	//br1->GetLeaf("pt")->SetLeafCount(br1->GetLeaf("njets"));		//<--
	br1->GetLeaf("pt")->SetAddress(JETS1.pt);
	
	TBranch *br2 = t->Branch("JETS2", &JETS2, "njets/I:et[njets]/F:pt[JETS2.njets]/F" );
	br2->GetLeaf("njets")->SetAddress(&JETS2.njets);	
	((TLeafI*) br2->GetLeaf("njets"))->SetMaximum(20);
	//br2->GetLeaf("et")->SetLeafCount(br2->GetLeaf("njets"));		//<--
	br2->GetLeaf("et")->SetAddress(JETS2.et);
	//br2->GetLeaf("pt")->SetLeafCount(br2->GetLeaf("njets"));		//<--
	br2->GetLeaf("pt")->SetAddress(JETS2.pt);
   
	for( size_t i = 0 ; i < 10 ; i++ ) {
		JETS1.njets = gRandom->Uniform(0,20);
		JETS2.njets = gRandom->Uniform(0,20);
		
		//printf("Event: %u, JETS1.njets=%d, JETS2.njets=%d\n", (unsigned int) i, JETS1.njets, JETS2.njets);
		
		for( size_t k = 0 ; k < 20 ; k++ ) {
			JETS1.et[k] = 0;
			JETS1.pt[k] = 0;
			JETS2.et[k] = -1;
			JETS2.pt[k] = 0;
		}
		
		for( size_t e = 0 ; e < JETS1.njets ; e++ ) {
			JETS1.et[e] = gRandom->Gaus(150,10);
			JETS1.pt[e] = gRandom->Gaus(100,10);
		}
		
		for( size_t e = 0 ; e < JETS2.njets ; e++ ) {
			JETS2.et[e] = gRandom->Gaus(150,10);
			JETS2.pt[e] = gRandom->Gaus(100,10);
		}		
		
		t->Fill();
	}
	
	t->SetScanField(50);
	t->Scan("JETS1.njets:JETS1.et:JETS1.pt:JETS2.njets:JETS2.et:JETS2.pt","", "colsize=10 col=:10.2f:10.2f::10.2f:10.2f", 2);
	
//	t->Write("",TObject::kOverwrite);
//	f->Close();
}

