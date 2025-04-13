{
// Fill out the code of the actual test
gSystem->Setenv("LINES","-1");
gROOT->ProcessLine(".L ptrptr.C");
gROOT->ProcessLine(".typedef");
}
