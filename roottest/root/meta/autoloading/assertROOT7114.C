int assertROOT7114() {
   gSystem->mkdir("someDir");
   gSystem->ChangeDirectory("someDir");
   int error = 0;
   if (gROOT->ProcessLine("MyROOT7114Class a", &error))
      return error;
   return 1;
}
