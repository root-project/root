void runprimary()
{
   gSystem->Load("libMatrix");
   TFile *f = TFile::Open("multi.zip#1");
   f->ls();
   TArchiveFile *a = f->GetArchive();
   a->GetMembers()->Print();
   TVectorD *gal = nullptr;
   f->GetObject("galaxy",gal);
   if (!gal) {
      cout << "Could not retrieve the object 'galaxy' from the zip archive!\n";
   } else if (gal->GetNrows() != 160801) {
      cout << "The retrieved TVectorD 'galaxy' has " << gal->GetNrows() << " rows instead of 160801\n";
   }
}
