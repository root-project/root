// void run(Int_t opt=0) {
{
   gROOT->ProcessLine(".! gmake -j2"); 
   
   if (pass) gSystem->Unload("vtest_C.so");
   //if (pass) gSystem->Unload("load_C.so");

   int result = gSystem->CompileMacro("load.C","k");
   if (result != 0 && result != 1) return;
   result = gSystem->CompileMacro("vtest.C","k");
   if (result != 0 && result != 1) return;

   if (opt) vtest();
   pass++;
}
