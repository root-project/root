{
// Fill out the code of the actual test
   TFile *_file0 = TFile::Open("nesting3.root");
   outTTree->Scan("GTT2.mvdrHits@.size()","","",10);
   outTTree->Scan("GTT2.mvdrHits.r","","",10);
   return 0;
}
