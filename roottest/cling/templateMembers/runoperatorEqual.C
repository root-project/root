{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L operatorEqual.C+");
#ifdef __ICLING__
gROOT->ProcessLine(".class StThreeVector");
#else
gROOT->ProcessLine(".Class StThreeVector<>");
#endif
}
