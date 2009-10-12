{
   TString roottest_home = gSystem->pwd();
      fprintf(stderr,"we are here\n");
   {
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
      gROOT->ProcessLine(Form(".L %sscripts/utils.cc+",roottest_home.Data()));
   }
   
}
