{
   // Fill out the code of the actual test
gROOT->ProcessLine(".x merging.C");
gROOT->ProcessLine(".x merging.C"); // Running it a second time used to lead to  core dumps.  This prevents the problem :)
}
