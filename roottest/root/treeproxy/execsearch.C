{
   TFile *f = TFile::Open("p0dNuESearch.root");
   TTree *t; f->GetObject("p0dNuESearch",t);

   // avoid extra message when default canvas is created
   // while such message can appear in random place relative to script ouput
   TCanvas c1("c0", "title", 500, 500);

   t->Process("searchSelector.h+");
}
