{
gROOT->ProcessLine(".O 0");
#ifdef ClingWorkAroundMissingAutoLoading
gSystem->Load("libTable");
#endif
gROOT->ProcessLine(".x testTable.C");
}
