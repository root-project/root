#include "OrderClass_v1.h"
#include "OrderClass.cxx"

#include "TFile.h"
#include "TTree.h"

void execWriteOrderClass(const char* filename = "orderClassTest.root")
{
   //	gSystem->Load("libMyClass_v1.so");
	
	TFile file(filename, "RECREATE");
   MyClass* myobj = new MyClass;

   file.WriteObject(myobj,"obj");

	TTree tree("testtree", "Tree with test objects");
	tree.Branch("myObjects", &myobj);

   myobj->addSomeData();
	
	tree.Fill();
	myobj->Print();
	
	file.Write();
}
