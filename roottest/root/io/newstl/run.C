// void run(Int_t opt=0) {
{
   if (opt/16 & 3 ) {
      gROOT->ProcessLine(".! gmake -j3");
   } else if (opt/16 & 2 ) {
      gROOT->ProcessLine(".! gmake -j3 deque"); 
   } else if (opt/16 & 1 ) {
      gROOT->ProcessLine(".! gmake -j3 vector"); 
   }
   
   if (pass) {
      gSystem->Unload("TestHelpers_C.so");
      pass = 0;
   }

   int result = gSystem->CompileMacro("TestHelpers.C","k");
   if (result != 0 && result != 1) return;
      
   if ( opt%16 & 2 ) {

      result = gSystem->CompileMacro("vectorHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("vtest.C","k");
      if (result != 0 && result != 1) return;

      if (opt/16) vtest();

   }

   if ( opt%16 & 2 ) {

      result = gSystem->CompileMacro("dequeHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("dtest.C","k");
      if (result != 0 && result != 1) return;

      if (opt/16) dtest();

   }

   pass++;
}
