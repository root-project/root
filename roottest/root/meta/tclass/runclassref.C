{
   // Avoid loading the library too soon
   gInterpreter->UnloadLibraryMap("RunMyClass_C");

   // Testing TClassRef.

   bool success = true;

   TClassRef clref("MyClass");
#ifdef __CINT__
   TClass *cl = clref.GetClass(); // CINT does not properly handle operator Cast
#else
   TClass *cl = clref;
#endif
   if (cl!=0) {
      success = false;
      cout << "Error: Ref to unknown class MyClass points to something (" << (void*)cl << ")\n";
   }
   TFile *f = TFile::Open("myclass.root");
#ifdef __CINT__
   cl = clref.GetClass(); // CINT does not properly handle operrator Cast
#else
   cl = clref;
#endif
   if (cl==0) {
      success = false;
      cout << "Error: Ref to emulated class MyClass did not find the TClass object\n";
   } else if (cl->IsLoaded()) {
      success = false;
      cout << "Error: Ref to emulated class MyClass thinks the class is loaded!\n";
   }
   TClass *cl2;
   gROOT->ProcessLine(".L RunMyClass.C+");
#ifdef __CINT__
   cl2 = clref.GetClass(); // CINT does not properly handle operrator Cast
#else
   cl2 = clref;
#endif
   if (cl2==0) {
      success = false;
      cout << "Error: Ref to compiled class MyClass did not find the TClass object\n";
   }
   if (cl==cl2) {
      success = false;
      cout << "Error: Ref to compiled class MyClass still points to the emulated TClass (which is now deleted!)\n";
      if (cl2->IsLoaded()) {
         cout << "Warning: but the class is apparently loaded anyway!?\n";
      }
   }
#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = !success;
#else
   return !success;
#endif
}
