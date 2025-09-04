{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L operatorEqual.C+");
#ifdef __CLING__
gROOT->ProcessLine(".class StThreeVector");
#else
gROOT->ProcessLine(".Class StThreeVector<>");
#endif
}
