/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// Example of a circular Tree
///
/// Circular Trees are interesting in online real time environments
/// to store the results of the last maxEntries events.
/// for more info, see TTree::SetCircular.
/// Circular trees must be memory resident.
///
/// \macro_code
///
/// \author Rene Brun

void circular() {
   auto T = new TTree("T","test circular buffers");
   TRandom r;
   Float_t px,py,pz;
   Double_t randomNum;
   UShort_t i;
   T->Branch("px",&px,"px/F");
   T->Branch("py",&py,"px/F");
   T->Branch("pz",&pz,"px/F");
   T->Branch("random",&randomNum,"random/D");
   T->Branch("i",&i,"i/s");
   T->SetCircular(20000); //keep a maximum of 20000 entries in memory
   for (i = 0; i < 65000; i++) {
      r.Rannor(px,py);
      pz = px*px + py*py;
      randomNum = r.Rndm();
      T->Fill();
   }
   T->Print();
}

