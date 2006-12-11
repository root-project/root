//Example of use of the TAlien class (an implementation of TGrid)
//Author: Andreas Peters
   
void alien()
{
   TString testdir = "root-test3";
   int nfiles      = 10;

   // connect to AliEn
   TGrid *alien = TGrid::Connect("alien", gSystem->Getenv("USER"), "",
                                 "-domain=cern.ch");
   if (alien->IsZombie()) {
      delete alien;
      return;
   }

   // get info on AliEn version
   printf("Using AliEn version %s\n", alien->GetInfo());

   // print current working directory
   printf("Current directory is %s\n", alien->Pwd());

   // check if directory exists
   Long_t size, flags, modtime;
   if (alien->GetPathInfo(testdir, &size, &flags, &modtime) == 0) {
      // delete existing directory
      alien->Rmdir(testdir);
   }

   // create a directory
   alien->Mkdir(testdir);

   // change directory
   alien->Cd(testdir);

   printf("Current directory is %s\n", alien->Pwd());

   // insert nfiles into the catalog
   Int_t i;
   char lfn[32], pfn[256];
   for (i = 0; i < nfiles; i++) {
      sprintf(lfn, "test-%d.root", i);
      sprintf(pfn, "rfio:/castor/cern.ch/user/r/rdm/mytest-%d.root", i);
      alien->AddFile(lfn, pfn, 1000000000);
   }

   // list the contents of a directory
   alien->ls("", "l");

   // get physical file name from lfn
   for (i = 0; i < nfiles; i++) {
      sprintf(lfn, "test-%d.root", i);
      char *pf = alien->GetPhysicalFileName(lfn);
      if (i == nfiles-1)
         printf("last pfn retrieved is: %s\n", pf);
      delete [] pf;
   }

   delete alien;
}

