// void run(Int_t opt=0) {
{
   gROOT->ProcessLine(".! gmake -j2"); 
   
   //if (pass) gSystem->Unload("vtest_C.so");
   //if (pass) gSystem->Unload("load_C.so");

   if (!gSystem->CompileMacro("load.C","k")) return;
   if (!gSystem->CompileMacro("vtest.C","k")) return;

   if (opt) vtest();

   pass ++;
}
