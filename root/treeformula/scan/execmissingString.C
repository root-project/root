{
   gErrorIgnoreLevel = kError;
   TFile *_file0 = TFile::Open("missingString.root");
   Long64_t res = t->Scan("fDetectorName","Length$(fDetectorName[0])>0","");
   res += t->Scan("fDetectorName","","");
   res += t->Scan("fSingleBoloSubRecs.fDetectorName","Length$(fDetectorName[0])>0","");
   res += t->Scan("fSingleBoloSubRecs.fDetectorName","","");
   return (res != 56);
}
