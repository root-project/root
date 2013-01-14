{
// Fill out the code of the actual test
gROOT->ProcessLine(".L templateKeyword.C+");
#ifdef ClingWorkAroundMissingAutoLoading
gROOT->ProcessLine("MyTest<double> d;");
#else
MyTest<double> d;
#endif
}
