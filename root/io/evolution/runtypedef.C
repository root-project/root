{
   gROOT->ProcessLine(".L typedefWrite.C+");
   TFile *f = new TFile("typedef.root");
   UHTTimeFitter *u;
   f->GetObject("myobject",&u);
}
