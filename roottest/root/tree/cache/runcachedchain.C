{
   auto c = new TChain("esdTree"); c->Add("AliESDs*.root");
   c->SetCacheSize();
   TTreeCache::SetLearnEntries(100);
   for (int i=0; i<10; ++i) {
      c->GetEntry(i);
   }
   TFile *curfile = c->GetCurrentFile(); 
   auto ca = (TTreeCache*)curfile->GetCacheRead();
   
   c->LoadTree(101);
   TBranch *b = c->GetBranch("AliESDRun.fMagneticField");
   Long64_t pos = b->GetBasketSeek(0);
   Int_t res = ca->ReadBuffer(0,pos,10);
   
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
