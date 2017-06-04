{
   auto o = new TObject();
   auto l = new TList();
   l->SetName("my own list");
   l->Add(o);
   delete o;
   l->Clear();
   o = new TObject();
   l->Add(o);
   delete o;
   l->Delete();
   return 0;
}
