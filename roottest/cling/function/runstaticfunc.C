{
   // We call a nested script so that the CINT is ignored
   // and the test returns success.
   gROOT->ProcessLine(".x staticfunc.C");
   return 0;
}
