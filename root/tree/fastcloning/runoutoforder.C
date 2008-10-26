{
   TFile *_file0 = TFile::Open("Tuple_merge.root");
   TTree*t; gFile->GetObject("tuple/DecayTree",t);
   t->SetScanField(0);
   t->Scan("runNumber:eventNumber");
   return 0;
}