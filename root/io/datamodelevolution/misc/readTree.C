
void readTree(const char* filename = "treeTest.root", bool loadSomeClass = true)
{
   gROOT->ProcessLine(".L MyClass.cxx+");
	if (loadSomeClass) gROOT->ProcessLine(".L SomeClass.cxx+");
	
	TFile file(filename, "READ");
   // TClass::GetClass("MyClass")->GetStreamerInfo()->ls();
	TTree* tree = (TTree*)file.Get("testtree");
	if (tree == NULL)
	{
		cerr << "ERROR: no tree found." << endl;
		return;
	}
	MyClass* myobj = new MyClass;
	tree->SetBranchAddress("myObjects", &myobj);
	
	tree->GetEvent(0);
   printf("Number of array elements: %d\n",myobj->Array().GetEntriesFast());
	myobj->Print();
}
