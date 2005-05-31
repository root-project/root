{
// Fill out the code of the actual test
   TFile *f = TFile::Open("multi.zip#1");
   f->ls();
   TArchiveFile *a = f->GetArchive();
   a->GetMembers()->Print();
   TVectorD *gal;
   f->GetObject("galaxy",gal);
   Bool_t result = kTRUE;
   if (gal==0) {
      cout << "Could not retrieve the object 'galaxy' from the zip archive!\n";
      result = kFALSE;
   } else if (gal->GetNrows()!=160801) {
      cout << "The retrieved TVectorD 'galaxy' has " << gal->GetNrows() << " rows instead of 160801\n";
      result = kFALSE;
   }
   return !result; // We need to return the correct error code for a shell script or makefile
}
