{
   TFile *f = TFile::Open("p0dNuESearch.root");
   TTree *t; f->GetObject("p0dNuESearch",t);
   t->MakeProxy("searchSelector","pns.C");
}
