{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L operatorEqual.C+");
gROOT->ProcessLine(".class StThreeVector<>");
}
