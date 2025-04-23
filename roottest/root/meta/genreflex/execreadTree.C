void execreadTree(const char *filename = "ofile.root")
{
   TFile file(filename, "READ");
   TTree *tree = (TTree *)file.Get("testtree");
   if (tree == NULL) {
      cerr << "ERROR: no tree found." << endl;
      return;
   }
   MyClass *myobj = new MyClass;
   tree->SetBranchAddress("myObjects", &myobj);

   tree->GetEvent(0);
   myobj->Print(nullptr);

   // Now, re-write the rootmap.
   const char* content = "[libMyClass_v1_dictrflx]\nclass MyClass";
   std::ofstream of ("al.rootmap");
   of << content << std::endl;

}
