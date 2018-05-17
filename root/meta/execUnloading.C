// Test Unloading
#include <iostream>

using std::cout;
using std::cerr;

void execUnloading(){
   // GLOBALS
   cout<< "Unloading of globals: VarDecl and EnumConstDecl.\n";
   TGlobal* g;
   gROOT->ProcessLine("int i;");
   g = (TGlobal*)gROOT->GetListOfGlobals()->FindObject("i");
   cout<< "Check validity before unloading, expected: true.\n";
   if (g->IsValid()) {
      cout << "<<i>> is valid.\n";
   } else {
      cerr << "<<i>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout<< "Check validity after unloading, expected: false.\n";
   if (!g->IsValid()) {
      cout << "<<i>> unloaded.\n";
   } else {
      cerr << "<<i>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine("int i;");
   cout<< "Check validity after reloading, expected: true.\n";
   if (g->IsValid()) {
      cout << "<<i>> is valid after reloading.\n";
   } else {
      cerr << "<<i>> is relaoded, but not valid.\n";
   }
   cout << "Check that the same obejct is used and not another created.\n";
   TGlobal* gReload = (TGlobal*)gROOT->GetListOfGlobals()->FindObject("i");
   if (g != gReload) {
      cerr << "The object " << g->GetName()
           << " is not reloaded, but created again!";
   }


   // ENUMS
   cout<< "\nUnloading of enum: EnumDecl and implicitly EnumConstDecl.\n";
   TEnum* e;
   gROOT->ProcessLine("enum MyEnum{c1,c2};");
   e = (TEnum*)gROOT->GetListOfEnums()->FindObject("MyEnum");
   cout << "Check validity before unloading, expected: true.\n";
   if (e->IsValid()) {
      cout << "<<MyEnum>> is valid.\n";
   } else {
      cerr << "<<MyEnum>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 2");
   // Verify the enum constants are invalidated from the list of globals.
   TGlobal* enumConstVar = (TGlobal*)gROOT->GetListOfGlobals()->FindObject("c1");
   if (enumConstVar != NULL) {
      cerr << "The unloaded enum constant " << enumConstVar->GetName()
           << "should not be in the list of globals.\n";
   }
   cout << "Check validity after unloading, expected: false.\n";
   if (!e->IsValid()) {
      cout << "<<MyEnum>> unloaded.\n";
   } else {
      cerr << "<<MyEnum>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine("enum MyEnum{c1,c2};");
   cout << "Check validity after reloading, expected: true.\n";
   if (e->IsValid()) {
      cout << "<<MyEnum>> is valid after reloading.\n";
   } else {
      cerr << "<<MyEnum>> is relaoded, but not valid.\n";
   }
   // Check that the same obejct is used and not another created.
   TEnum* eReload = (TEnum*)gROOT->GetListOfEnums()->FindObject("MyEnum");
   if (e != eReload) {
      cerr << "The object " << e->GetName()
           << "is not reloaded, but created again!\n";
   }


   // FUNCTIONS
   cout << "\nUnloading of functions: FunctionDecl.\n";
   TFunction* f;
   gROOT->ProcessLine(".rawInput 1");
   gROOT->ProcessLine("int func() { printf(\"This function is defined.\"); return 0;};");
   gROOT->ProcessLine(".rawInput 0");
   f = (TFunction*)gROOT->GetListOfGlobalFunctions()->FindObject("func");
   cout << "Check validity before unloading, expected: true.\n";
   if (f->IsValid()) {
      cout << "<<func>> is valid.\n";
   } else {
      cerr << "<<func>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check validity after unloading, expected: false.\n";
   if (!f->IsValid()) {
      cout << "<<func>> unloaded.\n";
   } else {
      cerr << "<<func>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine(".rawInput 1");
   gROOT->ProcessLine("int func() { printf(\"This function is defined.\"); return 0;};");
   gROOT->ProcessLine(".rawInput 0");
   cout << "Check validity after reloading, expected: true.\n";
   if (f->IsValid()) {
      cout << "<<func>> is valid after reloading.\n";
   } else {
      cerr << "<<func>> is relaoded, but not valid.\n";
   }
   // Check that the same obejct is used and not another created.
   TFunction* fReload = (TFunction*)gROOT->GetListOfGlobalFunctions()->FindObject("func");
   if (f != fReload) {
      cerr << "The object is not reloaded, but created again!\n";
   }

   // CLASSES
   cout << "\nUnloading of classes: RecordDecl.\n";
   gROOT->ProcessLine("class MyClass{ int data; int method(){return 0;} }");

   cout << "Check state of class, expected: \"3\" (kInterpreted).\n";
   cout << "The state of MyClass is: "
        << TClass::GetClass("MyClass")->GetState() << "\n";
   TDataMember* dm = (TDataMember*)TClass::GetClass("MyClass")->GetListOfDataMembers()->FindObject("data");
   cout << "Check validity before unloading, expected: true.\n";
   if (dm->IsValid()) {
      cout << "<<data>> is valid.\n";
   } else {
      cerr << "<<data>> is declared, but not valid.\n";
   }
   TMethod* m = (TMethod*)TClass::GetClass("MyClass")->GetListOfMethods()->FindObject("method");
   if (m->IsValid()) {
      cout << "<<method>> is valid.\n";
   } else {
      cerr << "<<method>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check state of class, expected: \"1\" (kForwardDeclared).\n";
   cout << "The state of MyClass is: "
        << TClass::GetClass("MyClass")->GetState() << "\n";
   cout << "Check validity after unloading, expected: false.\n";
   if (!dm->IsValid()) {
      cout << "<<data>> unloaded.\n";
   } else {
      cerr << "<<data>> is unloaded, but still valid.\n";
   }
   if (!m->IsValid()) {
      cout << "<<method>> unloaded.\n";
   } else {
      cerr << "<<method>> is unloaded, but still valid.\n";
   }
   TDataMember* unloadedDM = (TDataMember*)TClass::GetClass("MyClass")->GetListOfDataMembers()->FindObject("data");
   if (unloadedDM) {
      cerr << "The unloaded data member " << unloadedDM->GetName()
           << "should not be in the list of data members.\n";
   }
   TMethod* unloadedM = (TMethod*)TClass::GetClass("MyClass")->GetListOfMethods()->FindObject("method");
   if (unloadedM) {
      cerr << "The unloaded function member " << unloadedM->GetName()
           << "should not be in the list of data members.\n";
   }

   // NAMESPACES
   cout << "\nUnloading of namespaces: NamespaceDecl.\n";
   gROOT->ProcessLine("namespace ns{}");
   gROOT->ProcessLine("namespace ns{int k;}");
   TDataMember* ndm = (TDataMember*)TClass::GetClass("ns")->GetListOfDataMembers()->FindObject("k");
   cout << "Check validity before unloading, expected: true.\n";
   if (ndm->IsValid()) {
      cout << "<<ns::k>> is valid.\n";
   } else {
      cerr << "<<ns::k>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check validity after unloading, expected: false.\n";
   if (!ndm->IsValid()) {
      cout << "<<ns::k>> unloaded.\n";
   } else {
      cerr << "<<ns::k>> is unloaded, but still valid.\n";
   }
   TDataMember* unloadedNDM = (TDataMember*)TClass::GetClass("ns")->GetListOfDataMembers()->FindObject("k");
   if (unloadedNDM) {
      cerr << "The unloaded namespace data member " << unloadedNDM->GetName()
           << "should not be in the list of data members.\n";
   }

}
