//test TEnum and TEnumConstant

// MyEnumComment

enum EMyEnum {
   kMyEnumConstant = 42 // enumConstantComment
};

// Nested classes with enums.
class A
{
   public:
   class B
   {
      public:
      class C
      {
         public:
         enum enumC1 {
            kClassEnumC1A, // enum doc
            kClassEnumC1B
         };
         enum enumC2 {kClassEnumC2A,
                      kClassEnumC2B // enum doc, too
         };
         int x;
      };
      enum enumB {kClassEnumB1, // another enum doc
                  kClassEnumB2};
   };
   int y;
};

// Namespace enums.
namespace N {}

namespace N {
   enum namespaceEnum {
      kNamespace = 1 //  enum doc again
   };
}

int runTEnum()
{

   if (!(TEnum*)gROOT->GetListOfEnums()->FindObject("EMyEnum")) {
      Error("TEnum", "EMyEnum not found.");
   }

   // Find existing global TEnum in the list of enums
   if (!gROOT->GetListOfEnums()->FindObject("EProperty")) {
      Error("GetListOfEnums", "Did not find global enum %s in fEnums", "EProperty");
   }
   // Find existing global TEnumConstant in the list of globals
   if (!gROOT->GetListOfGlobals()->FindObject("kIsClass")) {
      Error("GetListOfGlobals", "Did not find global enum %s in fGlobals", "kIsClass");
   }
   // Assert that the constant is not in the list of globals
   if (gROOT->GetListOfGlobals()->FindObject("TString::kIgnoreCase")) {
      Error("GetListOfGlobals", "The following enum constant should not be in the list of globals: TString::kIgnoreCase");
   }
   // Assert that the constant is not in the list of globals, not even with unscoped name
   if (gROOT->GetListOfGlobals()->FindObject("kIgnoreCase")) {
      Error("GetListOfGlobals", "The following enum constant name should not be in the list of globals: kIgnoreCase");
   }
   
   // Find Existing TEnums in the TEnums list.
   if (TEnum* enumcaseCompare = (TEnum*)((TClass::GetClass("TString")->GetListOfEnums())->FindObject("ECaseCompare"))) { 
      const char* testEnum = "kIgnoreCase";
      const TEnumConstant* enumConst = enumcaseCompare->GetConstant(testEnum);
      if (!enumConst) {
         Error("TEnum::GetConstant", "The name of the constant is not correct.");
      } else {
         Long64_t enumValue = enumConst->GetValue();
         // Assert that the value of the constant is the same as the one in the list of Enums
         if (enumValue != TString::kIgnoreCase) {
            Error("TEnum::GetConstant", "The value int the list of enum constant is not the same as the one from the enum list.");
         }
      }
   }
   else {
      Error("GetListOfEnums", "The nested enum TString::ECaseCompare is not in the list of enums.");
   }

   //Assert that the list of constants of nested enums is created.
   TClass* t = TClass::GetClass("A::B::C");
   TList* enumsClassC = t->GetListOfEnums();
   printf("Enums in A::B::C:\n");
   enumsClassC->Print();
   //Assert that the enums of a nested class are found in the list of enums of that class.
   if (TEnum* enumC1Retrieved = (TEnum*)enumsClassC->FindObject("enumC1")) {
      enumC1Retrieved->GetConstants()->Print();
      // Assert that the constants are found in the list of constants of an TEnum.
      if (const TEnumConstant* nameConstant = enumC1Retrieved->GetConstant("kClassEnumC1A")) {
         //Assert that the value of the enum is correct.
         if(nameConstant->GetValue() != A::B::C::kClassEnumC1A) {
            Error("TEnum::GetConstant", "Value of the nested constant %s not correct.", "kClassEnumC1A");
         }
      } else {
         Error("TEnum::GetConstant", "Did not find nested enum name %s.", "kClassEnumC1");
      }
   } else {
      Error("GetListOfEnums", "Did not find enum.%s.", "A::B::C::");
   }
   
   // Assert that enums in namespaces are found.
   TClass* classN = TClass::GetClass("N", kTRUE);
   TList* enumsNamespaceN = classN->GetListOfEnums();
   printf("\n\nEnums in N:\n");
   enumsNamespaceN->Print();
   //Assert that the enums of namespace redecls are found in the list of enums of that namespace.
   if (TEnum* enumNRetrieved = (TEnum*)enumsNamespaceN->FindObject("namespaceEnum")) {
      enumNRetrieved->GetConstants()->Print();
      // Assert that the constants are found in the list of constants of an TEnum.
      if (const TEnumConstant* nameConstant = enumNRetrieved->GetConstant("kNamespace")) {
         //Assert that the value of the enum is correct.
         if(nameConstant->GetValue() != N::kNamespace) {
            Error("TEnum::GetConstant", "Value of the namespace constant %s not correct.", "kNamespace");
         }
      } else {
         Error("TEnum::GetConstant", "Did not find nested namespace enum name %s.", "kNamespace");
      }
   } else {
      Error("GetListOfEnums", "Did not find enum %s.", "namespaceEnum");
   }

   return 0;
}
