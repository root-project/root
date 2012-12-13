{
  TChain *tph;

   tph = new TChain("CBNT/t3333");
   tph->AddFile("file1.root");
   tph->AddFile("file2.root");

   tph->Merge("merged.root");

#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *fil; fil = new TFile("merged.root");
#else
   TFile *fil = new TFile("merged.root");
#endif
   fil->ls();
}
