{
gSystem->Setenv("LINES","-1");
gSystem->Load("libmaster");
gROOT->ProcessLine(".class Name::MyClass");
gSystem->Load("libslave1.so");
gROOT->ProcessLine(".class Name::MyClass");
gSystem->Load("libslave2.so");
gROOT->ProcessLine(".class Name::MyClass");
}
