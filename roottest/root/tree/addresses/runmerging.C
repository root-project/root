{
   // Fill out the code of the actual test
gROOT->ProcessLine(".x merging.C");
#if !defined ClingWorkAroundUnloadingVTABLES
gROOT->ProcessLine(".x merging.C"); // Running it a second time used to lead to  core dumps.  This prevents the problem :)
#else
gROOT->ProcessLine("merging()");
#endif
}
