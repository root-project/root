void TestClass(THtml& h, const char* exp, const char* clname) {
   TString act;
   h.GetModuleNameForClass(act, TClass::GetClass(clname));
   if (act != exp) {
      printf("FAIL: class %s should be in module %s but is in %s\n",
             clname, exp, act.Data());
   }
}

void runClassInModule() {
   gErrorIgnoreLevel = kWarning;
   THtml h;
   h.LoadAllLibs();
   h.SetInputDir("$ROOTSYS");
   h.MakeClass("INTENTIONALLY__BOGUS!");

   TestClass(h, "html", "THtml");
   TestClass(h, "core/base", "TObject");
   TestClass(h, "hist/hist", "TH1");
   TestClass(h, "hist/hist", "TH1F");
   TestClass(h, "math/mathcore", "TMath");
   TestClass(h, "core/base", "TParameter<double>");
   TestClass(h, "math/genvector","ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >");
   TestClass(h, "tmva", "TMVA::FitterBase");
   TestClass(h, "roofit/roofitcore", "RooAbsReal");
   TestClass(h, "roofit/roostats", "RooStats::PointSetInterval");
   TestClass(h, "math/matrix", "TMatrixTSparse<double>");
}
