{
gSystem->Load("myclbad");
gROOT->ProcessLine(".typedef MyClass<Toy>::value_type");
gROOT->ProcessLine("gROOT->GetClass(\"MyClass<Toy>::value_type\")");
}
