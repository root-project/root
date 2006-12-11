// Example to write & read a Tree built with a complex class inheritance tree.
// It demonstrates usage of inheritance and TClonesArrays
// This is simplied / stripped extract of an event structure used within the
// Marabou project (http://www.bl.physik.uni-muenchen.de/marabou/html/)
//
//to run this example, do:
// root > .x clonesA_Event.C

void clonesA_Event_w()
{
// protect against old ROOT versions
   if ( gROOT->GetVersionInt() < 30503 ) {
      cout << "Works only with ROOT version >= 3.05/03" << endl;
      return;
   }
   if ( gROOT->GetVersionDate() < 20030406 ) {
      cout << "Works only with ROOT CVS version after 5. 4. 2003" << endl;
      return;
   }

   //write a Tree
   TFile *hfile = new TFile("clonesA_Event.root","RECREATE","Test TClonesArray");
   TTree *tree  = new TTree("clonesA_Event","An example of a ROOT tree");
   TUsrSevtData1 *event1 = new TUsrSevtData1();
   TUsrSevtData2 *event2 = new TUsrSevtData2();
   tree->Branch("top1","TUsrSevtData1",&event1,8000,99);
   tree->Branch("top2","TUsrSevtData2",&event2,8000,99);
   for (Int_t ev = 0; ev < 10; ev++) {
      cout << "event " << ev << endl;
      event1->SetEvent(ev);
      event2->SetEvent(ev);
      tree->Fill();
      if (ev <3) tree->Show(ev);
   }
   tree->Write();
   tree->Print();
   delete hfile;
}
 
void clonesA_Event_r()
{
   //read the Tree
   TFile * hfile = new TFile("clonesA_Event.root");
   TTree *tree = (TTree*)hfile->Get("clonesA_Event");

   TUsrSevtData1 * event1 = 0;
   TUsrSevtData2 * event2 = 0;
   tree->SetBranchAddress("top1",&event1);
   tree->SetBranchAddress("top2",&event2);
   for (Int_t ev = 0; ev < 8; ev++) {
      tree->Show(ev);
      cout << "Pileup event1: " <<  event1->GetPileup() << endl;
      cout << "Pileup event2: " <<  event2->GetPileup() << endl;
      event1->Clear();
      event2->Clear();
 //     gObjectTable->Print();          // detect possible memory leaks
   }
   delete hfile;
}
 
void clonesA_Event() {
   gROOT->ProcessLine(".L clonesA_Event.cxx+");  // compile shared lib
   clonesA_Event_w();                            // write the tree
   clonesA_Event_r();                            // read back the tree
}
