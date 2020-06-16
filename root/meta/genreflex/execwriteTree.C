#include <fstream>

void execwriteTree(const char *filename = "ofile.root")
{

   TFile file(filename, "RECREATE");
   TTree tree("testtree", "Tree with test objects");
   MyClass *myobj = new MyClass;
   tree.Branch("myObjects", &myobj);

   myobj->addSomeData();

   tree.Fill();
   myobj->Print();

   file.Write();

   // Now, re-write the rootmap.
   const char* content = "[libMyClass_v2_dictrflx]\nclass MyClass";
   std::ofstream of ("al.rootmap");
   of << content << std::endl;

}
