{
#ifdef ClingWorkAroundMissingImplicitAuto
   TChain *
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
      c ;
#endif
#endif
   c = new TChain("esdTree"); c->Add("AliESDs*.root");
   c->SetCacheSize();
   TTreeCache::SetLearnEntries(100);
   for (int i=0; i<10; ++i) {
      c->GetEntry(i);
   }
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *curfile; curfile = c->GetCurrentFile(); 
#else
   TFile *curfile = c->GetCurrentFile(); 
#endif
#ifdef ClingWorkAroundMissingImplicitAuto
   TTreeCache *
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
      ca ;
#endif
#endif
   ca = (TTreeCache*)curfile->GetCacheRead(); 
   
   
   c->LoadTree(101);
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TBranch *b; b = c->GetBranch("AliESDRun.fMagneticField");
   Long64_t pos ; pos = b->GetBasketSeek(0);
   Int_t res ; res = ca->ReadBuffer(0,pos,10);
#else
   TBranch *b = c->GetBranch("AliESDRun.fMagneticField");
   Long64_t pos = b->GetBasketSeek(0);
   Int_t res = ca->ReadBuffer(0,pos,10);
#endif
   
   if (res != 1) {
      fprintf(stdout,"ERROR: Could not find the basket bytes for the first basket of AliESDRun.fMagneticField in the cache\n");
#ifdef ClingWorkAroundBrokenUnnamedReturn
      int result1 =  1;
   }
   int result2 = 0;
#else
   return 1;
}
return 0;
#endif
}
