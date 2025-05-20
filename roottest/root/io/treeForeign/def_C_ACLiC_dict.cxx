// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME CcLfIUsersfIfabiofIForkedROOTfIrootfIroottestfIrootfIiofItreeForeignfIdef_C_ACLiC_dict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

// Header files passed as explicit arguments
#include "C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.C"

// Header files passed via #pragma extra_include

namespace std {
   namespace ROOTDict {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *std_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("std", 0 /*version*/, "vcruntime_new.h", 25,
                     ::ROOT::Internal::DefineBehavior((void*)nullptr,(void*)nullptr),
                     &std_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_DICT_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_DICT_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *std_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}

namespace ROOT {
   static TClass *MyClass_Dictionary();
   static void MyClass_TClassManip(TClass*);
   static void *new_MyClass(void *p = nullptr);
   static void *newArray_MyClass(Long_t size, void *p);
   static void delete_MyClass(void *p);
   static void deleteArray_MyClass(void *p);
   static void destruct_MyClass(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::MyClass*)
   {
      ::MyClass *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::MyClass));
      static ::ROOT::TGenericClassInfo 
         instance("MyClass", "", 3,
                  typeid(::MyClass), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &MyClass_Dictionary, isa_proxy, 4,
                  sizeof(::MyClass) );
      instance.SetNew(&new_MyClass);
      instance.SetNewArray(&newArray_MyClass);
      instance.SetDelete(&delete_MyClass);
      instance.SetDeleteArray(&deleteArray_MyClass);
      instance.SetDestructor(&destruct_MyClass);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::MyClass*)
   {
      return GenerateInitInstanceLocal(static_cast<::MyClass*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::MyClass*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *MyClass_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal(static_cast<const ::MyClass*>(nullptr))->GetClass();
      MyClass_TClassManip(theClass);
   return theClass;
   }

   static void MyClass_TClassManip(TClass* theClass){
      theClass->CreateAttributeMap();
      TDictAttributeMap* attrMap( theClass->GetAttributeMap() );
      attrMap->AddProperty("file_name","C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.h");
   }

} // end of namespace ROOT

namespace ROOT {
   static void *new_Wrapper(void *p = nullptr);
   static void *newArray_Wrapper(Long_t size, void *p);
   static void delete_Wrapper(void *p);
   static void deleteArray_Wrapper(void *p);
   static void destruct_Wrapper(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::Wrapper*)
   {
      ::Wrapper *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::Wrapper >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("Wrapper", ::Wrapper::Class_Version(), "", 10,
                  typeid(::Wrapper), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::Wrapper::Dictionary, isa_proxy, 4,
                  sizeof(::Wrapper) );
      instance.SetNew(&new_Wrapper);
      instance.SetNewArray(&newArray_Wrapper);
      instance.SetDelete(&delete_Wrapper);
      instance.SetDeleteArray(&deleteArray_Wrapper);
      instance.SetDestructor(&destruct_Wrapper);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::Wrapper*)
   {
      return GenerateInitInstanceLocal(static_cast<::Wrapper*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::Wrapper*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr Wrapper::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *Wrapper::Class_Name()
{
   return "Wrapper";
}

//______________________________________________________________________________
const char *Wrapper::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::Wrapper*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int Wrapper::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::Wrapper*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *Wrapper::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::Wrapper*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *Wrapper::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::Wrapper*)nullptr)->GetClass(); }
   return fgIsA;
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_MyClass(void *p) {
      return  p ? new(p) ::MyClass : new ::MyClass;
   }
   static void *newArray_MyClass(Long_t nElements, void *p) {
      return p ? new(p) ::MyClass[nElements] : new ::MyClass[nElements];
   }
   // Wrapper around operator delete
   static void delete_MyClass(void *p) {
      delete (static_cast<::MyClass*>(p));
   }
   static void deleteArray_MyClass(void *p) {
      delete [] (static_cast<::MyClass*>(p));
   }
   static void destruct_MyClass(void *p) {
      typedef ::MyClass current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::MyClass

//______________________________________________________________________________
void Wrapper::Streamer(TBuffer &R__b)
{
   // Stream an object of class Wrapper.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(Wrapper::Class(),this);
   } else {
      R__b.WriteClassBuffer(Wrapper::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_Wrapper(void *p) {
      return  p ? new(p) ::Wrapper : new ::Wrapper;
   }
   static void *newArray_Wrapper(Long_t nElements, void *p) {
      return p ? new(p) ::Wrapper[nElements] : new ::Wrapper[nElements];
   }
   // Wrapper around operator delete
   static void delete_Wrapper(void *p) {
      delete (static_cast<::Wrapper*>(p));
   }
   static void deleteArray_Wrapper(void *p) {
      delete [] (static_cast<::Wrapper*>(p));
   }
   static void destruct_Wrapper(void *p) {
      typedef ::Wrapper current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::Wrapper

namespace ROOT {
   // Registration Schema evolution read functions
   int RecordReadRules_def_C_ACLiC_dict() {
      return 0;
   }
   static int _R__UNIQUE_DICT_(ReadRules_def_C_ACLiC_dict) = RecordReadRules_def_C_ACLiC_dict();R__UseDummy(_R__UNIQUE_DICT_(ReadRules_def_C_ACLiC_dict));
} // namespace ROOT
namespace {
  void TriggerDictionaryInitialization_def_C_ACLiC_dict_Impl() {
    static const char* headers[] = {
"C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.C",
nullptr
    };
    static const char* includePaths[] = {
"C:/Users/fabio/ForkedROOT/build/include",
"C:/Users/fabio/ForkedROOT/build/include/",
"C:/Users/fabio/ForkedROOT/build/roottest/root/io/treeForeign/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "def_C_ACLiC_dict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
)DICTFWDDCLS"
R"DICTFWDDCLS(#pragma clang diagnostic ignored "-Wignored-attributes"
)DICTFWDDCLS"
R"DICTFWDDCLS(#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
)DICTFWDDCLS"
R"DICTFWDDCLS(extern int __Cling_AutoLoading_Map;
)DICTFWDDCLS"
R"DICTFWDDCLS(class __attribute__((annotate("$clingAutoload$C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.C")))  MyClass;
)DICTFWDDCLS"
R"DICTFWDDCLS(class __attribute__((annotate("$clingAutoload$C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.C")))  Wrapper;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "def_C_ACLiC_dict dictionary payload"

#ifndef R__ACLIC_ROOTMAP
  #define R__ACLIC_ROOTMAP 1
#endif
#ifndef __ACLIC__
  #define __ACLIC__ 1
#endif

#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "C:/Users/fabio/ForkedROOT/root/roottest/root/io/treeForeign/def.C"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"", payloadCode, "@",
"MyClass", payloadCode, "@",
"Wrapper", payloadCode, "@",
"Wrapper::fgIsA", payloadCode, "@",
"def", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("def_C_ACLiC_dict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_def_C_ACLiC_dict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_def_C_ACLiC_dict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_def_C_ACLiC_dict() {
  TriggerDictionaryInitialization_def_C_ACLiC_dict_Impl();
}
