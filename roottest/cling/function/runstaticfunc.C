{
   // We call a nested script so that the CLING is ignored
   // and the test returns success.
   gROOT->ProcessLine(".x staticfunc.C");
   return 0;
}
