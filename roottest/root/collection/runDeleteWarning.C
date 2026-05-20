{
   if (ROOT::Internal::DeleteChangesMemory()) {
      // The internal testing for after deletion use is disable so we have to fake it.
      cout << "Error in <TList::Clear>: A list is accessing an object (0x00000000) already deleted (list name = my own list)\n";
      cout << "Error in <TList::Delete>: A list is accessing an object (0x00000000) already deleted (list name = my own list)\n";
   } else {
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
   }
   return 0;
}
