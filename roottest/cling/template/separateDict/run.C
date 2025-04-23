{
gSystem->Setenv("LINES","-1");
gSystem->Load("libmaster");
gROOT->ProcessLine(".Class Name::MyClass");
gSystem->Load("libslave1");
gROOT->ProcessLine(".Class Name::MyClass");
gSystem->Load("libslave2");
gROOT->ProcessLine(".Class Name::MyClass");
}
