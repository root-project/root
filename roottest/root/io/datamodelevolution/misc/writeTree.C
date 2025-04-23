
void writeTree(const char* filename = "treeTest.root")
{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L MyClass.cxx+");
   gROOT->ProcessLine(".L SomeClass.cxx+");
#endif

   TFile file(filename, "RECREATE");
   TTree tree("testtree", "Tree with test objects");
   MyClass* myobj = new MyClass;
   tree.Branch("myObjects", &myobj);
	
   myobj->Add(new TNamed("aaa", "AAAAAAAA"));
   myobj->Add(new SomeClass("bbb"));
   myobj->Add((TObject*)1);
   myobj->Add(new TNamed("ccc", "CCCCCCCC"));
   myobj->Add(new SomeClass("ddd"));
   myobj->Array()[2] = 0;
   //gDebug = 7;
   tree.Fill();
   //gDebug = 0;
   printf("Number of array elements: %d\n",myobj->Array().GetEntriesFast());
   // myobj->Array().ls();
   myobj->Print();
	
   file.Write();
}
