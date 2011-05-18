//  Example of a circular Tree
//  Circular Trees are interesting in online real time environments
//  to store the results of the last maxEntries events.
//  for more info, see TTree::SetCircular
// Author: Rene Brun
void circular() {
   gROOT->cd(); //make sure that the Tree is memory resident
   TTree *T = new TTree("T","test circular buffers");
   TRandom r;
   Float_t px,py,pz;
   Double_t random;
   UShort_t i;
   T->Branch("px",&px,"px/F");
   T->Branch("py",&py,"px/F");
   T->Branch("pz",&pz,"px/F");
   T->Branch("random",&random,"random/D");
   T->Branch("i",&i,"i/s");
   T->SetCircular(20000); //keep a maximum of 20000 entries in memory
   for (i = 0; i < 65000; i++) {
      r.Rannor(px,py);
      pz = px*px + py*py;
      random = r.Rndm();
      T->Fill();
   }
   T->Print();
}

