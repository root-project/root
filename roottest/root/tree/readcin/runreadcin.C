{
  TTree* t = new TTree("test","test");
  t->ReadFile("test.csv","",':');
  t->Print();
  return 0;
}
