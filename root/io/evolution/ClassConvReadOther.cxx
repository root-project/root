{
   gROOT->ProcessLine(".L ClassConvOther.cxx+");
   TFile *f = new TFile("ClassConvNew.root","READ");
   TopLevel *l= 0;
   f->GetObject("MyTopLevel",l);
   cout << "TopLevel value is " << l->GetValue() << endl;
   f->Close();
}
