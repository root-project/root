{ 
gSystem->Setenv("LINES","-1");
gSystem->Load("libmaster");
gROOT->ProcessLine(".class Master::Container");
gSystem->Load("libslave1");
gROOT->ProcessLine(".class Master::Container");
gSystem->Load("libslave2");
gROOT->ProcessLine(".class Master::Container");
}
