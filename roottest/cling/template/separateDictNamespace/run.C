{
gSystem->Setenv("LINES","-1");
gSystem->Load("libmasterSeparateNS");
gROOT->ProcessLine(".Class Master::Container");
gSystem->Load("libslave1SeparateNS");
gROOT->ProcessLine(".Class Master::Container");
gSystem->Load("libslave2SeparateNS");
gROOT->ProcessLine(".Class Master::Container");
}
