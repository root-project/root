{
#ifndef ClingWorkAroundNoDotOptimization
gROOT->ProcessLine(".O 0");
#endif
#ifdef ClingWorkAroundMissingAutoLoading
gSystem->Load("libTable");
#endif
gROOT->ProcessLine(".x testTable.C");
}
