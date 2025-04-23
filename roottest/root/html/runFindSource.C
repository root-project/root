void FindAny(bool found, const TString& fsname, TClass* cl, TString baseexpected, const char* ext, const char* tag) {

   if (baseexpected == "") {
      baseexpected = cl->GetName();
      Ssiz_t posCol = baseexpected.First('<');
      if (posCol != -1) {
         baseexpected.Remove(posCol, baseexpected.Length());
      }
      posCol = baseexpected.Last(':');
      if (posCol != -1) {
         baseexpected.Remove(0, posCol + 1);
      }
      baseexpected += ext;
   }

   if (!found) {
      if (baseexpected != "FAIL") {
         printf("FAIL: %s file for class %s not found\n", tag, cl->GetName());
      }
      return;
   } else {
      if (baseexpected == "FAIL") {
         printf("FAIL: expected to not find %s file for class %s but got %s\n",
                tag, cl->GetName(), fsname.Data());
         return;
      }
   }
   if (!fsname.EndsWith(baseexpected)) {
      printf("FAIL: class %s expected %s file %s, got %s\n",
             cl->GetName(), tag, baseexpected.Data(), fsname.Data());
      return;
   }
}

void Find(THtml& h, TClass* cl, const char* hdr = "", const char* src = "") {
   TString fsname;
   Bool_t found = h.GetDeclFileName(cl, 1, fsname);
   FindAny(found, fsname, cl, hdr, ".h", "declaration");

   found = h.GetImplFileName(cl, 1, fsname);
   FindAny(found, fsname, cl, src, ".cxx", "implementation"); 
}

void runFindSource() {
   gErrorIgnoreLevel = kWarning;
   THtml h;
   h.SetInputDir("$ROOTSYS");

   Find(h, TObject::Class());
   Find(h, TH1::Class());
   Find(h, TH1F::Class(), "TH1.h", "TH1.cxx");

   Find(h, TClass::GetClass("TMath"));

   Find(h, TParameter<double>::Class());

   TClass* cl = TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >");
   Find(h, cl, "", "FAIL");

   Find(h, TClass::GetClass("TMVA::FitterBase"));

   Find(h, TClass::GetClass("RooAbsReal"));
   Find(h, TClass::GetClass("RooStats::PointSetInterval"));
   Find(h, TClass::GetClass("TMatrixTSparse<double>"));
}
