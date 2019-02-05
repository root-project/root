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
      cerr << "ERROR: <<i>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout<< "Check validity after unloading, expected: false.\n";
   if (!g->IsValid()) {
      cout << "<<i>> unloaded.\n";
   } else {
      cerr << "ERROR: <<i>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine("int i;");
   cout<< "Check validity after reloading, expected: true.\n";
   if (g->IsValid()) {
      cout << "<<i>> is valid after reloading.\n";
   } else {
      cerr << "ERROR: <<i>> is reloaded, but not valid.\n";
   }
   cout << "Check that the same obejct is used and not another created.\n";
   TGlobal* gReload = (TGlobal*)gROOT->GetListOfGlobals()->FindObject("i");
   if (g != gReload) {
      cerr << "ERROR: The object " << g->GetName()
           << " is not reloaded, but created again!";
   }

   // ENUMS
   cout<< "\nUnloading of enum: EnumDecl and implicitly EnumConstDecl.\n";

   // Trigger autoparsing now, not between enum decl and unloading!
   TClass::GetClass("TEnumConstant")->GetClassInfo();

   gROOT->ProcessLine("enum MyEnum{c1,c2};");
   TEnum* e = (TEnum*)gROOT->GetListOfEnums()->FindObject("MyEnum");
   cout << "Check validity before unloading, expected: true.\n";
   if (e->IsValid()) {
      cout << "<<MyEnum>> is valid.\n";
   } else {
      cerr << "ERROR: <<MyEnum>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   // Verify the enum constants are invalidated from the list of globals.
   TGlobal* enumConstVar = (TGlobal*)gROOT->GetListOfGlobals()->FindObject("c1");
   if (enumConstVar != NULL) {
      cerr << "ERROR: The unloaded enum constant " << enumConstVar->GetName()
           << " should not be in the list of globals.\n";
   }
   cout << "Check validity after unloading, expected: false.\n";
   if (!e->IsValid()) {
      cout << "<<MyEnum>> unloaded.\n";
   } else {
      cerr << "ERROR: <<MyEnum>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine("enum MyEnum{c1,c2};");
   cout << "Check validity after reloading, expected: true.\n";
   if (e->IsValid()) {
      cout << "<<MyEnum>> is valid after reloading.\n";
   } else {
      cerr << "ERROR: <<MyEnum>> is reloaded, but not valid.\n";
   }
   // Check that the same obejct is used and not another created.
   TEnum* eReload = (TEnum*)gROOT->GetListOfEnums()->FindObject("MyEnum");
   if (e != eReload) {
      cerr << "ERROR: The object " << e->GetName()
           << " is not reloaded, but created again!\n";
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
      cerr << "ERROR: <<func>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check validity after unloading, expected: false.\n";
   if (!f->IsValid()) {
      cout << "<<func>> unloaded.\n";
   } else {
      cerr << "ERROR: <<func>> is unloaded, but still valid.\n";
   }
   // Reload the object.
   gROOT->ProcessLine(".rawInput 1");
   gROOT->ProcessLine("int func() { printf(\"This function is defined.\"); return 0;};");
   gROOT->ProcessLine(".rawInput 0");
   cout << "Check validity after reloading, expected: true.\n";
   if (f->IsValid()) {
      cout << "<<func>> is valid after reloading.\n";
   } else {
      cerr << "ERROR: <<func>> is relaoded, but not valid.\n";
   }
   // Check that the same obejct is used and not another created.
   TFunction* fReload = (TFunction*)gROOT->GetListOfGlobalFunctions()->FindObject("func");
   if (f != fReload) {
      cerr << "ERROR: The object is not reloaded, but created again!\n";
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
      cerr << "ERROR: <<data>> is declared, but not valid.\n";
   }
   TMethod* m = (TMethod*)TClass::GetClass("MyClass")->GetListOfMethods()->FindObject("method");
   if (m->IsValid()) {
      cout << "<<method>> is valid.\n";
   } else {
      cerr << "ERROR: <<method>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check state of class, expected: \"1\" (kForwardDeclared).\n";
   cout << "The state of MyClass is: "
        << TClass::GetClass("MyClass")->GetState() << "\n";
   cout << "Check validity after unloading, expected: false.\n";
   if (!dm->IsValid()) {
      cout << "<<data>> unloaded.\n";
   } else {
      cerr << "ERROR: <<data>> is unloaded, but still valid.\n";
   }
   if (!m->IsValid()) {
      cout << "<<method>> unloaded.\n";
   } else {
      cerr << "ERROR: <<method>> is unloaded, but still valid.\n";
   }
   TDataMember* unloadedDM = (TDataMember*)TClass::GetClass("MyClass")->GetListOfDataMembers()->FindObject("data");
   if (unloadedDM) {
      cerr << "ERROR: The unloaded data member " << unloadedDM->GetName()
           << " should not be in the list of data members.\n";
   }
   TMethod* unloadedM = (TMethod*)TClass::GetClass("MyClass")->GetListOfMethods()->FindObject("method");
   if (unloadedM) {
      cerr << "ERROR: The unloaded function member " << unloadedM->GetName()
           << " should not be in the list of data members.\n";
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
      cerr << "ERROR: <<ns::k>> is declared, but not valid.\n";
   }
   gROOT->ProcessLine(".undo 1");
   cout << "Check validity after unloading, expected: false.\n";
   if (!ndm->IsValid()) {
      cout << "<<ns::k>> unloaded.\n";
   } else {
      cerr << "ERROR: <<ns::k>> is unloaded, but still valid.\n";
   }
   TDataMember* unloadedNDM = (TDataMember*)TClass::GetClass("ns")->GetListOfDataMembers()->FindObject("k");
   if (unloadedNDM) {
      cerr << "ERROR: The unloaded namespace data member " << unloadedNDM->GetName()
           << " should not be in the list of data members.\n";
   }

}
