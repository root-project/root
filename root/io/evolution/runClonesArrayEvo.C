{
   #include "Event_3.cxx+"
   TFile *f = TFile::Open("Event_2.root");
   TTree *t; f->GetObject("T",t);
   t->GetEntry(0);
   return 0;
}
