{
   gErrorIgnoreLevel = kError;
   TFile *_file0 = TFile::Open("missingString.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *t; _file0->GetObject("t",t);
#endif
   Long64_t res = t->Scan("fDetectorName","Length$(fDetectorName[0])>0","");
   res += t->Scan("fDetectorName","","");
   res += t->Scan("fSingleBoloSubRecs.fDetectorName","Length$(fDetectorName[0])>0","");
   res += t->Scan("fSingleBoloSubRecs.fDetectorName","","");
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(res != 56);
#else
   return (res != 56);
#endif
}
