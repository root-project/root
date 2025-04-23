{
   gROOT->ProcessLine(".L ClassConvOther.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
   "TFile *f = new TFile(\"ClassConvNew.root\",\"READ\");"
   "TopLevel *l= 0;"
   "f->GetObject(\"MyTopLevel\",l);"
   "cout << \"TopLevel value is \" << l->GetValue() << endl;"
   "f->Close();");
#else
   TFile *f = new TFile("ClassConvNew.root","READ");
   TopLevel *l= 0;
   f->GetObject("MyTopLevel",l);
   cout << "TopLevel value is " << l->GetValue() << endl;
   f->Close();
#endif
}
