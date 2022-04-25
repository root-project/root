{ 
gSystem->Setenv("LINES","-1");
gSystem->Load("libmaster");
gROOT->ProcessLine(".Class Master::Container");
gSystem->Load("libslave1");
gROOT->ProcessLine(".Class Master::Container");
gSystem->Load("libslave2");
gROOT->ProcessLine(".Class Master::Container");
}
