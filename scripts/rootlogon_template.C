{
   TString roottest_home = gSystem->pwd();
   {
      Int_t pos = roottest_home.Index("/roottest/");
      if (pos==TString::kNPOS) {
         pos = roottest_home.Index("\\roottest\\");
         if (pos!=TString::kNPOS) pos += strlen("\\roottest\\");
      } else {
         pos += strlen("/roottest/");
      }
      if (pos!=TString::kNPOS) {
         roottest_home.Remove(pos);
      }
      gROOT->ProcessLine(Form(".L %sscripts/utils.cc+",roottest_home.Data()));
   }
   
}
