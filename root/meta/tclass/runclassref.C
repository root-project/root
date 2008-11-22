{
   // Avoid loading the library too soon
   gInterpreter->UnloadLibraryMap("RunMyClass_C");

   // Testing TClassRef.

   bool success = true;

   TClassRef ref("MyClass");
#ifdef __CINT__
   TClass *cl = ref.GetClass(); // CINT does not properly handle operrator Cast
#else
   TClass *cl = ref;
#endif
   if (cl!=0) {
      success = false;
      cout << "Ref to unknown class MyClass points to something (" << (void*)cl << ")\n";
   }
   TFile *f = TFile::Open("myclass.root");
#ifdef __CINT__
   cl = ref.GetClass(); // CINT does not properly handle operrator Cast
#else
   cl = ref;
#endif
   if (cl==0) {
      success = false;
      cout << "Ref to emulated class MyClass did not find the TClass object\n";
   }
   TClass *cl2;
   gROOT->ProcessLine(".L RunMyClass.C+");
#ifdef __CINT__
   cl2 = ref.GetClass(); // CINT does not properly handle operrator Cast
#else
   cl2 = ref;
#endif
   if (cl2==0) {
      success = false;
      cout << "Ref to compiled class MyClass did not find the TClass object\n";
   }
   if (cl==cl2) {
      success = false;
      cout << "Ref to compiled class MyClass still points to the emulated TClass (which is now deleted!)\n";
   }
   return !success;
}
