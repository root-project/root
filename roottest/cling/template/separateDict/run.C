{
gSystem->Setenv("LINES","-1");
gSystem->Load("libmasterSeparate");
gROOT->ProcessLine(".Class Name::MyClass");
gSystem->Load("libslave1Separate");
gROOT->ProcessLine(".Class Name::MyClass");
gSystem->Load("libslave2Separate");
gROOT->ProcessLine(".Class Name::MyClass");
}
