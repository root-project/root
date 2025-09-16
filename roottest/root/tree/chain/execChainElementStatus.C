
void AddFiles(TChain &ch) {
   ch.AddFile("st4.root");
   ch.AddFile("st-empty.root");
   ch.AddFile("st2.root");
   ch.AddFile("st-doesnotexist.root");
   ch.AddFile("st-notree.root");
   ch.AddFile("st8.root");
}

void Print(TChain &ch)
{
   TIter next(ch.GetListOfFiles());
   TChainElement *element{nullptr};
   while( (element = (TChainElement*)next()) ) {
      cout << "Filename:   " << element->GetTitle() << '\n';
      cout << "Entries:    " << element->GetEntries() << '\n';
      cout << "LoadResult: " << element->GetLoadResult() << '\n';
      cout << '\n';
      // element->Dump();
   }
   cout << endl;
}

void testGetEntries() {
   TChain ch("tester");
   AddFiles(ch);

   // The test is meant to traverse the entire dataset to populate
   // the information of each TChainElement correctly.
   ch.LoadTree(TTree::kMaxEntries - 1);

   Print(ch);

}

void testDraw() {
   TChain ch("tester");
   AddFiles(ch);

   ch.Draw("value");

   Print(ch);
}

void execChainElementStatus() {
   testGetEntries();
   testDraw();
}