{
gROOT->ProcessLine(".L template.so");
TClass * c = gROOT->GetClass("MyTemplate<const int*>");
c->Dump();
c = gROOT->GetClass("MyTemplate<const double*>");
c->Dump();
}
