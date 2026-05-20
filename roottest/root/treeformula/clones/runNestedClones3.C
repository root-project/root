{
// Fill out the code of the actual test
   TFile *_file0 = TFile::Open("nesting3.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *outTTree; _file0->GetObject("outTTree",outTTree);
#endif
#ifdef ClingWorkAroundCallfuncAndInline
   outTTree->Scan("GTT2.mvdrHits@.GetEntries()","","",10);
#else
   outTTree->Scan("GTT2.mvdrHits@.size()","","",10);
#endif
   outTTree->Scan("GTT2.mvdrHits.r","","",10);
#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}
