void tree3() {
// Example of a Tree where branches are variable length arrays
// Run this script with 
//   .x tree3.C
// then in a new session, use, eg the TreeViewer to histogram
// the ntuple entries
//  root > TFile f("tree3.root")
//  root > t3->StartViewer()
//
//  Author: Rene Brun
      
   const Int_t kMaxTrack = 500;
   Int_t ntrack;
   Int_t stat[kMaxTrack]; 
   Int_t sign[kMaxTrack]; 
   Float_t px[kMaxTrack]; 
   Float_t py[kMaxTrack]; 
   Float_t pz[kMaxTrack]; 
   Float_t zv[kMaxTrack];
   Float_t chi2[kMaxTrack];
  
   TFile f("tree3.root","recreate");
   TTree *t3 = new TTree("t3","Reconst ntuple");
   t3->Branch("ntrack",&ntrack,"ntrack/I");
   t3->Branch("stat",stat,"stat[ntrack]/I");
   t3->Branch("sign",sign,"sign[ntrack]/I");
   t3->Branch("px",px,"px[ntrack]/F");
   t3->Branch("py",py,"py[ntrack]/F");
   t3->Branch("pz",pz,"pz[ntrack]/F");
   t3->Branch("zv",zv,"zv[ntrack]/F");
   t3->Branch("chi2",chi2,"chi2[ntrack]/F");

   for (Int_t i=0;i<1000;i++) {
      Int_t nt = gRandom->Rndm()*(kMaxTrack-1);
      ntrack = nt;
      for (Int_t n=0;n<nt;n++) {
         stat[n] = n%3;
         sign[n] = i%2;
         px[n]   = gRandom->Gaus(0,1);
         py[n]   = gRandom->Gaus(0,2);
         pz[n]   = gRandom->Gaus(10,5);
         zv[n]   = gRandom->Gaus(100,2);
         chi2[n] = gRandom->Gaus(0,.01);
      }
      t3->Fill();
   }   
   t3->Print();
   t3->Write();
}
      
