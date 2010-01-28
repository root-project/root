{
   TString roottest_home = gSystem->Getenv("ROOTTEST_HOME");
   if (roottest_home == "") {
      roottest_home = gSystem->pwd();
      Int_t pos = roottest_home.Index("/roottest/");
      if (pos==TString::kNPOS) {
         pos = roottest_home.Index("\\roottest\\");
         if (pos!=TString::kNPOS) {
            Int_t next = pos;
            while (next != TString::kNPOS) {
               pos = next;
               next = roottest_home.Index("\\roottest\\",pos+1);
            }
            pos += strlen("\\roottest\\");
         } else {
            printf("ERROR in rootlogon.C: cannot determine ROOTTEST_HOME!\n"
                   "Please rename the roottest root directory, it should be called \"roottest\"\n");
            exit(1);
         }
      } else {
         Int_t next = pos;
         while (next != TString::kNPOS) {
            pos = next;
            next = roottest_home.Index("/roottest/",pos+1);
         }
         pos += strlen("/roottest/");
      }
      if (pos!=TString::kNPOS) {
         roottest_home.Remove(pos);
      }
   }
   if (!roottest_home.EndsWith("/")) {
      roottest_home += "/";
   }
   {
      gROOT->ProcessLine(Form(".L %sscripts/utils.cc+",roottest_home.Data()));   
   }
}
