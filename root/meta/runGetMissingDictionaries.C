#include <TNamed.h>
#include <TClass.h>
#include <THashTable.h>
#include <TError.h>
#include <map>
#include <vector>
#include <iostream>

using namespace std;

namespace BasicTests {
struct NoDict {};
typedef NoDict NoDict_t;
typedef NoDict_t NoDict1_t;
typedef NoDict1_t NoDict2_t;
typedef NoDict2_t NoDict3_t;
struct NoDictTypdefs {
   NoDict fPlain;
   NoDict_t fTD0;
   NoDict1_t fTD1;
   NoDict2_t fTD2;
   NoDict3_t fTD3;
};

struct HasVecDouble32 {
   std::map<UShort_t, Double32_t> fMem;
   //std::vector<Double32_t> vDo32;
};

typedef std::map<UShort_t, Double32_t> Vec32_t;
typedef Double32_t Double32_too_t;
struct HasVecDoubleTD32 {
   Vec32_t fMem;
   std::map<UShort_t, Double32_too_t> fMem2;
};

struct TestAll {
   NoDictTypdefs fOne;
   HasVecDouble32 fTwo;
   HasVecDoubleTD32 fThree;
};
}

class NoA;
struct NoDictClass {
   NoA* fa;
};
class ArrayTry;
struct Base;
struct MemberHidden;
#ifndef __ROOTCLING__
struct Base {
   float Member;
};
struct MemberHidden: public Base {
   float Member;
};
#endif


class Member: public TNamed {
public:
   virtual ~Member() {};
};

typedef Member Member_t;
typedef Member MemArray_t[3];
typedef const Member MemCArray_t[3];
typedef MemArray_t* MemArrayP_t;
typedef const MemCArray_t* MemCArrayP_t;

class TypedefExample{};
typedef TypedefExample TypedefExample_t;
typedef TypedefExample_t TypedefExample_t_t;

template <typename T>
class Tmplt {
   ArrayTry* fT[12];
   std::vector<T> fVecT;
   T* fPtrT;

};

class TmpParam {};
class TmpTmpParam {};
class ExtraTmp {};

class ParamL1;
class ParamL2;

template <class T> class Inner { std::vector<T> fValues; };
template <class T> class Outer { Inner<T> fValues; };

template <class T, class S> class ParamList {
   T* fparmT[10];
   std::vector<S*> fparmS;
   S* sPtr;
};

struct TestClass {
   TestClass(): fMemCArray() {}
   BasicTests::TestAll fBasicTests;
   NoDictClass* fNoDict;
   TNamed* fNamedPtr;
   TNamed fNamed;
   int fPOD;
   TestClass* fInfiniteRecursion;
   Member_t fMember;
   MemArray_t fMemArray;
   MemArray_t* fMemArray_Ptr;
   MemArrayP_t fMemArrayP;
   MemArrayP_t* fMemArrayP_Ptr;
   MemCArray_t fMemCArray;
   MemCArrayP_t fMemCCArray;
   const MemCArrayP_t* fMemCCArray_CPtr;
   Tmplt<int> fTmpltInt;
   Tmplt<Tmplt<NoDictClass*> > fTmpltTmpltNoDict;
   Tmplt<Member> fTmpltMember;
   MemberHidden* fMemberHidden;
   Outer<Double32_t> obj;
   Tmplt<TmpParam> templateParam;
   Tmplt<Tmplt<TmpTmpParam> > tempWithTempParam;
   Tmplt<Tmplt<Tmplt<ExtraTmp> > > extraTmp;
   ParamList<ParamL1, ParamL2> pramList;
   TypedefExample_t ftypedefsCheck;
};


int runGetMissingDictionaries()
{
   // Method to assert the dictionaries.

   TClass* myClass = TClass::GetClass("TestClass");
   if (myClass->HasDictionary())
      Error("TClass::HasDictionary", "The class %s does not have a dictionary.", "TestClass");

   THashTable expectedResult;
   // Hard coded expected results.
   expectedResult.Add(TClass::GetClass("NoDictClass"));
   expectedResult.Add(TClass::GetClass("TestClass"));
   expectedResult.Add(TClass::GetClass("MemberHidden"));
   expectedResult.Add(TClass::GetClass("Tmplt<Tmplt<NoDictClass*> >"));
   expectedResult.Add(TClass::GetClass("Tmplt<int>"));
   expectedResult.Add(TClass::GetClass("Member"));
   expectedResult.Add(TClass::GetClass("Tmplt<Member>"));
   expectedResult.Add(TClass::GetClass("Outer<Double32_t>"));
   expectedResult.Add(TClass::GetClass("Tmplt<TmpParam>"));
   expectedResult.Add(TClass::GetClass("Tmplt<Tmplt<TmpTmpParam> >"));
   expectedResult.Add(TClass::GetClass("Tmplt<Tmplt<Tmplt<ExtraTmp> > >"));
   expectedResult.Add(TClass::GetClass("ParamList<ParamL1,ParamL2>"));
   expectedResult.Add(TClass::GetClass("TypedefExample"));
   expectedResult.Add(TClass::GetClass("BasicTests::TestAll"));
   expectedResult.Add(TClass::GetClass("Base"));

   cerr<<"No recursion:"<<endl; // Write on the same stream of the errors below
   THashTable missingDictClassesNoRecursion;
   // Assert GetMissingDictionaries without recursion.
   myClass->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   //missingDictClassesNoRecursion.Print();
   if (!missingDictClassesNoRecursion.IsEmpty()) {
      if (missingDictClassesNoRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesNoRecursion.GetEntries());
      }
      TIterator* it = missingDictClassesNoRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesNoRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "The set of missing classes is not created");
   }


   // Assert GetMissingDictionaries with recursion.
   // Hard code expected results with recursion.
   expectedResult.Add(TClass::GetClass("ArrayTry"));
   expectedResult.Add(TClass::GetClass("NoA"));
   expectedResult.Add(TClass::GetClass("vector<Tmplt<NoDictClass*> >"));
   expectedResult.Add(TClass::GetClass("Tmplt<NoDictClass*>"));
   expectedResult.Add(TClass::GetClass("vector<Member>"));
   expectedResult.Add(TClass::GetClass("Inner<Double32_t>"));
   expectedResult.Add(TClass::GetClass("TmpParam"));
   expectedResult.Add(TClass::GetClass("TmpTmpParam"));
   expectedResult.Add(TClass::GetClass("Tmplt<TmpTmpParam>"));
   expectedResult.Add(TClass::GetClass("ExtraTmp"));
   expectedResult.Add(TClass::GetClass("Tmplt<ExtraTmp>"));
   expectedResult.Add(TClass::GetClass("Tmplt<Tmplt<ExtraTmp> >"));
   expectedResult.Add(TClass::GetClass("vector<Tmplt<Tmplt<ExtraTmp> > >"));
   expectedResult.Add(TClass::GetClass("vector<NoDictClass*> "));
   expectedResult.Add(TClass::GetClass("vector<TmpTmpParam>"));
   expectedResult.Add(TClass::GetClass("vector<Tmplt<ExtraTmp> >"));
   expectedResult.Add(TClass::GetClass("vector<ExtraTmp> "));
   expectedResult.Add(TClass::GetClass("vector<Tmplt<TmpTmpParam> >"));
   expectedResult.Add(TClass::GetClass("vector<TmpParam>"));
   expectedResult.Add(TClass::GetClass("vector<Double32_t>"));
   expectedResult.Add(TClass::GetClass("ParamL1"));
   expectedResult.Add(TClass::GetClass("ParamL2"));
   expectedResult.Add(TClass::GetClass("vector<ParamL2*>"));
   expectedResult.Add(TClass::GetClass("BasicTests::NoDictTypdefs"));
   expectedResult.Add(TClass::GetClass("BasicTests::NoDict"));
   expectedResult.Add(TClass::GetClass("BasicTests::HasVecDoubleTD32"));
   expectedResult.Add(TClass::GetClass("BasicTests::HasVecDouble32"));
   expectedResult.Add(TClass::GetClass("map<unsigned short,Double32_t>"));

   cerr<<"With recursion:"<<endl; // Write on the same stream of the errors below
   THashTable missingDictClassesRecursion;
   myClass->GetMissingDictionaries(missingDictClassesRecursion, true);
   //missingDictClassesRecursion.Print();
   if (!missingDictClassesRecursion.IsEmpty()) {
      if (missingDictClassesRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "The set of missing classes is not created");
   }

   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   TClass *uq = TClass::GetClass("std::unique_ptr<TObject>");
   uq->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries without recursion", uq->GetName());
      missingDictClassesNoRecursion.ls();
   }
   uq->GetMissingDictionaries(missingDictClassesRecursion, true);
   if (!missingDictClassesRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries", uq->GetName());
      missingDictClassesRecursion.ls();
   }

   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   uq = TClass::GetClass("std::unique_ptr<NoDictClass>");
   expectedResult.Add(TClass::GetClass("NoDictClass"));
   uq->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      if (missingDictClassesNoRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesNoRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesNoRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesNoRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty without recursion", uq->GetName());
      missingDictClassesNoRecursion.ls();
   }
   uq->GetMissingDictionaries(missingDictClassesRecursion, true);
   expectedResult.Add(TClass::GetClass("NoA"));
   if (!missingDictClassesRecursion.Empty()) {
      if (missingDictClassesRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty ", uq->GetName());
      missingDictClassesRecursion.ls();
   }

   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   TClass *ar = TClass::GetClass("std::array<TObject, 4>");
   ar->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries without recursion", ar->GetName());
      missingDictClassesNoRecursion.ls();
   }
   ar->GetMissingDictionaries(missingDictClassesRecursion, true);
   if (!missingDictClassesRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries", ar->GetName());
      missingDictClassesRecursion.ls();
   }

   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   ar = TClass::GetClass("std::array<long, 4>");
   ar->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries without recursion", ar->GetName());
      missingDictClassesNoRecursion.ls();
   }
   ar->GetMissingDictionaries(missingDictClassesRecursion, true);
   if (!missingDictClassesRecursion.Empty()) {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly some missing dictionaries", ar->GetName());
      missingDictClassesRecursion.ls();
   }


   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   ar = TClass::GetClass("std::array<NoDictClass, 4>");
   expectedResult.Add(TClass::GetClass("NoDictClass"));
   ar->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      if (missingDictClassesNoRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesNoRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesNoRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesNoRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty without recursion", ar->GetName());
      missingDictClassesNoRecursion.ls();
   }
   expectedResult.Add(TClass::GetClass("NoA"));
   ar->GetMissingDictionaries(missingDictClassesRecursion, true);
   if (!missingDictClassesRecursion.Empty()) {
      if (missingDictClassesRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty ", ar->GetName());
      missingDictClassesRecursion.ls();
   }

   expectedResult.Clear();
   missingDictClassesNoRecursion.Clear();
   missingDictClassesRecursion.Clear();

   TClass *tp = TClass::GetClass("std::tuple<NoDictClass, int, TObject>");
   expectedResult.Add(TClass::GetClass("NoDictClass"));
   tp->GetMissingDictionaries(missingDictClassesNoRecursion, false);
   if (!missingDictClassesNoRecursion.Empty()) {
      if (missingDictClassesNoRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesNoRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesNoRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesNoRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty without recursion", tp->GetName());
      missingDictClassesNoRecursion.ls();
   }
   expectedResult.Add(TClass::GetClass("NoA"));
   tp->GetMissingDictionaries(missingDictClassesRecursion, true);
   if (!missingDictClassesRecursion.Empty()) {
      if (missingDictClassesRecursion.GetEntries() != expectedResult.GetEntries()) {
         Error("TClass::GetMissingClassDictionaries", "The set of classes with missing dictionaries does not contain the correct number of elements (expected: %d got %d).",expectedResult.GetEntries(),missingDictClassesRecursion.GetEntries());
         //         expectedResult.ls();
         //         missingDictClassesRecursion.ls();
      }
      TIterator* it = missingDictClassesRecursion.MakeIterator();
      TClass* cl = 0;
      while ((cl = (TClass*)it->Next())) {
         if (!expectedResult.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s is not in the expected set.", cl->GetName());
         }
      }
      it = expectedResult.MakeIterator();
      while ((cl = (TClass*)it->Next())) {
         if (!missingDictClassesRecursion.FindObject(cl)) {
            Error("TCling::GetMissingDictionaries", "Class %s with no dictionaries is not in the set.", cl->GetName());
         }
      }
   } else {
      Error("TClass::GetMissingClassDictionaries", "For %s found unexpectedly empty ", ar->GetName());
      missingDictClassesRecursion.ls();
   }


   return 0;
}
