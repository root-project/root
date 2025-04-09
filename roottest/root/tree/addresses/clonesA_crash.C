{
   if (gROOT->GetClass("TUsrSevtData1")==0) gROOT->ProcessLine(".L clonesA_Event.cxx+g");  // compile shared lib
   gROOT->ProcessLine(".L listadd.C");
   TFile *hfile = new TFile("clonesA_Event.root","RECREATE","Test TClonesArray");
   TTree *tree  = new TTree("clonesA_Event","An example of a ROOT tree");
   TUsrSevtData1 *event1 = new TUsrSevtData1();
   TBranchElement *b1 = (TBranchElement*) tree->Branch("top1","TUsrSevtData1",&event1,8000,99);
   TBranchElement *b2 = (TBranchElement*) tree->Branch("top2.","TUsrSevtData1",&event1,8000,99);
 
cerr << "Event      ptr: " << &event1 << endl;
cerr << "Event      add: " <<  event1 << endl;
cerr << "fHitBuffer add: " <<  event1->GetHitBuffer() << endl;
cerr << "fHits      add: " <<  event1->GetHitBuffer()->GetCA() << endl;
cerr << endl;
cerr << "Event br   add: " <<  (void*) b1->GetObject() << endl;
cerr << "fHitBuffer.fHits br obj: " << (void*) ((TBranchElement*)tree->GetBranch("fHitBuffer.fHits"))->GetObject() << endl;
cerr << "fHitBuffer.fHits.fUniqueID br obj: " << (void*) ((TBranchElement*)tree->GetBranch("fHitBuffer.fHits.fUniqueID"))->GetObject() << endl;
}
