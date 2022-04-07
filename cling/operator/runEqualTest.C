{
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L equal.C+");
gROOT->ProcessLine(".class privateOp2");
}
