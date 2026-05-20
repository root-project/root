{
  TTree* t = new TTree("test","test");
  t->ReadStream(std::cin,"",':');
  t->Print();
  return 0;
}
