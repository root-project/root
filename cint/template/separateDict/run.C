{
gSystem->Setenv("LINES","-1");
gSystem->Load("libmaster");
gROOT->ProcessLine(".class Name::MyClass");
gSystem->Load("libslave1");
gROOT->ProcessLine(".class Name::MyClass");
gSystem->Load("libslave2");
gROOT->ProcessLine(".class Name::MyClass");
}
