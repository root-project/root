{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L operatorEqual.C+");
gROOT->ProcessLine(".Class StThreeVector<>");
}
