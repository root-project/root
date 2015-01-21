#include "OrderClass_v2.h"
#include "OrderClass.cxx"

#include "TFile.h"
#include "TTree.h"

void execReadOrderClass(const char* filename = "orderClassTest.root")
{
   //	gSystem->Load("libMyClass_v2.so");
	
	TFile file(filename, "READ");
	TTree* tree = (TTree*)file.Get("testtree");
	if (tree == NULL)
	{
		cerr << "ERROR: no tree found." << endl;
		return;
	}
	MyClass* myobj = new MyClass;
	tree->SetBranchAddress("myObjects", &myobj);
	
	tree->GetEvent(0);
	myobj->Print();
}
