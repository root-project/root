{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L operatorEqual.C+");
#ifdef __CINT__
gROOT->ProcessLine(".class StThreeVector");
#else
gROOT->ProcessLine(".Class StThreeVector<>");
#endif
}
