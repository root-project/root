{
gSystem->Load("myclgood");
gROOT->ProcessLine(".typedef MyClass<Toy>::value_type");
if (gROOT->GetClass("MyClass<Toy>::value_type") == 0) 
  fprintf(stdout,"Error MyClass<Toy>::value_type  not retrieveable by gROOT::GetClass\n");
}
