{
   gROOT->ProcessLine(".L ClassConvOld.cxx+");
   TFile *f = new TFile("ClassConv.root","READ");
   TopLevel *l= 0;
   f->GetObject("MyTopLevel",l);
   cout << "TopLevel value is " << l->GetValue() << endl;
   f->Close();
}