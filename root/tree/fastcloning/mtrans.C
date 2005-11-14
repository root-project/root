void mtrans(char *filename) 
{
   TChain *c = new TChain("T");
   c->AddFile(Form("one/%s",filename));
   c->AddFile(Form("two/%s",filename));
   c->Merge(Form("merge/%s",filename));
}
