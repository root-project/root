//Example to write & read a Tree built with a complex class inheritance tree
//to run this example, do:
// root > .x otto.C

void ottow()
{
   //write a Tree
   TFile *hfile = new TFile("otto.root","RECREATE","Test TClonesArray");
   TTree *tree  = new TTree("otto","An example of a ROOT tree");
   TUsrSevtData2 *event = new TUsrSevtData2();
   tree->Branch("top","TUsrSevtData2",&event,8000,99);
   for (Int_t ev = 0; ev < 10; ev++) {
      cout << "event " << ev << endl;
      event->SetEvent(ev);
      tree->Fill();
      if (ev <3) tree->Show(ev);
   }
   tree->Write();
   tree->Print();
   delete hfile;
}
 
void ottor()
{
   //read the Tree
   TFile * hfile = new TFile("otto.root");
   TTree *tree = (TTree*)hfile->Get("otto");

   TUsrSevtData2 * event = 0;
   tree->SetBranchAddress("top",&event);
   for (Int_t ev = 0; ev < 3; ev++) {
      tree->Show(ev);
      event->Clear();
   }
   delete hfile;
}
 
void otto() {
   gROOT->ProcessLine(".L otto.cxx+");
   ottow();
   ottor();
}
