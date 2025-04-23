{
  gSystem->Load("libtransientNoctor_dictrflx");
  TClass::GetClass("MyClass")->GetStreamerInfo()->ls();
}
