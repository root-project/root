void alien()
{
   TGrid *alien = TGrid::Connect("alien", gSystem->Getenv("USER"), "", "-domain=cern.ch");
   if (alien->IsZombie()) {
      delete alien;
      return;
   }

   printf("Current directory is %s\n", alien->Pwd());

   Long_t size, flags, modtime;
   if (alien->GetPathInfo("root-test", &size, &flags, &modtime) == 0)
      alien->Rmdir("root-test");

   alien->Mkdir("root-test");

   alien->Cd("root-test");

   printf("Current directory is %s\n", alien->Pwd());

   Int_t i;
   char lfn[32], pfn[256];
   for (i = 0; i < 10; i++) {
      sprintf(lfn, "test-%d.root", i);
      sprintf(pfn, "rfio:/castor/cern.ch/user/r/rdm/mytest-%d.root", i);
      alien->AddFile(lfn, pfn, 1000000000);
   }

   alien->ls("", "l");

   delete alien;
}

