#if defined(ClingWorkAroundBrokenUnnamedReturn)
void runmergeWithDir()
#endif
{
  TChain *tph;
   tph = new TChain("CBNT/t3333");
   tph->AddFile("file1.root");
   tph->AddFile("file2.root");
   tph->Merge("merged.root");
   TFile *fil = new TFile("merged.root");
   fil->ls();
}
