{
gSystem->Load("libPhysics");
gROOT->ProcessLine(".L jet.C+g");
writeJet();
readJet();
}
