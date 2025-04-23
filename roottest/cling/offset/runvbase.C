#include "TInterpreter.h"
#include "TClass.h"
#include <iostream>

using std::cerr;
using std::endl;

// Inheritance structure to test.
struct Fill {
   char S[1024];
};

// Virtual base class.
struct Top {
  int fValue;
  virtual ~Top() {}
};

struct Mid1 : public virtual Top
{
   int fMid1;
};

struct Mid2 : public virtual Top
{
   int fMid2;
};

struct Bottom : public Mid1, Mid2
{
   int fBottom;
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4584)
#endif
#ifdef __CLING__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winaccessible-base"
#endif

struct Basement: public Fill, public Mid2, public Bottom {
   char FillMore[17];
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __CLING__
#pragma clang diagnostic pop
#endif

extern "C" int printf(const char*,...);

// Compiler computed offset.
template <typename DERIVED, typename TARGET>
long CompOffset(const DERIVED* obj) {
   char* addrDerived = (char*) obj;
   // NOTE: this casts even unrelated types, yielding a 0 offset!
   char* addrBase = (char*)((const TARGET*)obj);
   // [base1 base2 base3]
   // ^-- obj starts here
   //                   ^-- obj + sizeof(obj)
   //              ^-- offset to base3
   return addrBase - addrDerived;
}

// Compiler computed offset.
template <typename DERIVED, typename TARGET>
long CompOffsetNotDerived(const TARGET* obj) {
   char* addrDerived = (char*) dynamic_cast<const DERIVED*>((const TARGET*)obj);
   // NOTE: this casts even unrelated types, yielding a 0 offset!
   char* addrBase = (char*) obj;
   // [base1 base2 base3]
   // ^-- obj starts here
   //                   ^-- obj + sizeof(obj)
   //              ^-- offset to base3
   return addrBase-addrDerived;
}

// Interpreter computes offset.
long InterpOffsetTClassInterface(void* obj, const char* fromDerivedClassName, const char* toBaseClassName, bool isDerivedObject = true) {
   TClass* clDerived = TClass::GetClass(fromDerivedClassName);
   TClass* clBase = TClass::GetClass(toBaseClassName);
   return clDerived->GetBaseClassOffset(clBase, obj, isDerivedObject);
}

long InterpOffsetTClassInfoInterface(void* obj, const char* fromDerivedClassName, const char* toBaseClassName, bool isDerivedObject = true) {
   ClassInfo_t* cliDerived = gInterpreter->ClassInfo_Factory(fromDerivedClassName);
   ClassInfo_t* cliTarget = gInterpreter->ClassInfo_Factory(toBaseClassName);
   long offset = -1;
   offset = gInterpreter->ClassInfo_GetBaseOffset(cliDerived, cliTarget, obj, isDerivedObject);
   gInterpreter->ClassInfo_Delete(cliDerived);
   gInterpreter->ClassInfo_Delete(cliTarget);
   return offset;
}

template <typename DERIVED, typename TARGET>
void CheckFor(DERIVED* obj,
              const char* fromDerivedClassName, const char* toBaseClassName) {
   auto compilerOffset = CompOffset<DERIVED, TARGET>(obj);
   auto interpreterOffset = InterpOffsetTClassInterface(obj, fromDerivedClassName, toBaseClassName);
   if(compilerOffset == interpreterOffset) {
      printf("derived %s -> base %s: Compiler and interpreter say the same value.\n",
             fromDerivedClassName, toBaseClassName);
   } else if (interpreterOffset == -1) {
      printf("derived %s -> base %s: Compiler says something, interpreter says -1.\n",
             fromDerivedClassName, toBaseClassName);
   } else {
      printf("derived %s -> base %s: Compiler (%ld) says different value than interpreter (%ld).\n",
             fromDerivedClassName, toBaseClassName, compilerOffset, interpreterOffset);
   }
}

template <typename DERIVED, typename TARGET>
void CheckForNotDerived(TARGET* obj,
              const char* fromDerivedClassName, const char* toBaseClassName) {
   long offsetComp = CompOffsetNotDerived<DERIVED, TARGET>(obj);
   long offsetTClass = InterpOffsetTClassInterface(obj, fromDerivedClassName,
                                                  toBaseClassName, false);
   if (!strcmp(fromDerivedClassName, "Basement")
       && !strcmp(toBaseClassName, "Top")) {
         // On some platforms (or stdlibs?) the interpreted dynamic_cast from a
         // virtual base to derived returns 0, and the offset calculation thus returns
         // -1.
     if (offsetTClass == offsetComp || offsetTClass == -1) {
        printf("derived %s -> base %s: Compiler and TClass agree or offset fail - this is as good as it gets.\n",
               fromDerivedClassName, toBaseClassName);
     } else {
        printf("derived %s -> base %s: Compiler (%ld) and TClass (%ld) disagree!\n",
               fromDerivedClassName, toBaseClassName, offsetComp, offsetTClass);
     }
   } else if (offsetTClass == offsetComp) {
      printf("derived %s -> base %s: Compiler and TClass say the same value.\n",
             fromDerivedClassName, toBaseClassName);
   } else {
      printf("derived %s -> base %s: Compiler (%ld) says different value than TClass (%ld).\n",
             fromDerivedClassName, toBaseClassName, offsetComp, offsetTClass);
   }
}

template <typename DERIVED, typename TARGET>
void CheckForWithClassInfo(DERIVED* obj,
              const char* fromDerivedClassName, const char* toBaseClassName) {
   auto compilerOffset = CompOffset<DERIVED, TARGET>(obj);
   auto interpreterOffset = InterpOffsetTClassInterface(obj, fromDerivedClassName, toBaseClassName);
   if(compilerOffset == interpreterOffset) {
      printf("derived %s -> base %s: Compiler and interpreter say the same value.\n",
             fromDerivedClassName, toBaseClassName);
   } else if (interpreterOffset == -1) {
      printf("derived %s -> base %s: Compiler says something, interpreter says -1.\n",
             fromDerivedClassName, toBaseClassName);
   } else {
      printf("derived %s -> base %s: Compiler (%ld) says different value than interpreter (%ld).\n",
             fromDerivedClassName, toBaseClassName, compilerOffset, interpreterOffset);
   }
}

void runvbase() {
   Basement *obj = new Basement;
   Top *baseObj = obj;

   CheckFor<Basement, Top>(obj, "Basement", "Top");
   // Check for the caching of the function pointer in TClingClassInfo.
   CheckFor<Basement, Top>(obj, "Basement", "Top");
   printf("Top does not derive from Basement:\n");
   CheckFor<Basement, Top>(obj, "Top", "Basement");
   printf("The object is a base ptr object:\n");
   CheckForNotDerived<Basement, Top>(baseObj, "Basement", "Top");
   CheckFor<Basement, Fill>(obj, "Basement", "Fill");
   CheckFor<Basement, Mid1>(obj, "Basement", "Mid1");
   // This will result in an error from the Compiler function first.
   // Ambiguous:
   //   struct Basement -> struct Mid2
   //   struct Basement -> struct Bottom -> struct Mid2
   // Also see check for error at the end.
   //CheckFor<Basement, Mid2>(obj, "Basement", "Mid2");
   CheckFor<Basement, Bottom>(obj, "Basement", "Bottom");
   // Basement doesn't derive from Basement, but this should still be
   // handled (offset 0)
   CheckFor<Basement, Basement>(obj, "Basement", "Basement");

   CheckFor<Bottom, Top>(obj, "Bottom", "Top");
   CheckFor<Bottom, Mid1>(obj, "Bottom", "Mid1");
   CheckFor<Bottom, Mid2>(obj, "Bottom", "Mid2");
   CheckFor<Bottom, Bottom>(obj, "Bottom", "Bottom");

   CheckFor<Mid1, Top>(obj, "Mid1", "Top");
   CheckFor<Mid1, Mid1>(obj, "Mid1", "Mid1");
   // Classes are unrelated so should return -1.
   cerr << "The derived class does not derive from base, thus we expect "
      "different results from compiler and cling.\n";
   CheckFor<Mid1, Mid2>(obj, "Mid1", "Mid2");

   //to Bottom or to Mid2, did we not already check for Bottom?
   // Need to cast to Bottom or ambiguous:
   //  struct Basement -> struct Mid2
   //  struct Basement -> struct Bottom -> struct Mid2
   CheckFor<Mid2, Top>((Bottom*)obj, "Mid2", "Top");
   CheckFor<Mid2, Mid2>((Bottom*)obj, "Mid2", "Mid2");

   // Assert that this cannot be determined, i.e. that after this call
   // Root survives and complains about the ambiguous cast:
   //  struct Basement -> struct Mid2
   //  struct Basement -> struct Bottom -> struct Mid2
   cerr << "Multiple paths case:\n";
   Int_t ambiguousOffset = InterpOffsetTClassInterface(obj, "Basement", "Mid2");

   cerr << "derived Basement -> base Mid2: TClass says "
        << ambiguousOffset << '\n';

   // No multiple generations of the same function when not using the TClass interface
   CheckForWithClassInfo<Basement, Top>(obj, "Basement", "Top");
   // Check for the caching of the function pointer in TClingClassInfo.
   CheckForWithClassInfo<Basement, Top>(obj, "Basement", "Top");
   printf("Top does not derive from Basement:\n");
   CheckForWithClassInfo<Basement, Top>(obj, "Top", "Basement");
   CheckForWithClassInfo<Basement, Fill>(obj, "Basement", "Fill");
   CheckForWithClassInfo<Basement, Mid1>(obj, "Basement", "Mid1");
   // This will result in an error from the Compiler function first.
   // Ambiguous:
   //   struct Basement -> struct Mid2
   //   struct Basement -> struct Bottom -> struct Mid2
   // Also see check for error at the end.
   //CheckForWithClassInfo<Basement, Mid2>(obj, "Basement", "Mid2");
   CheckForWithClassInfo<Basement, Bottom>(obj, "Basement", "Bottom");
   // Basement doesn't derive from Basement, but this should still be
   // handled (offset 0)
   CheckForWithClassInfo<Basement, Basement>(obj, "Basement", "Basement");

   CheckForWithClassInfo<Bottom, Top>(obj, "Bottom", "Top");
   CheckForWithClassInfo<Bottom, Mid1>(obj, "Bottom", "Mid1");
   CheckForWithClassInfo<Bottom, Mid2>(obj, "Bottom", "Mid2");
   CheckForWithClassInfo<Bottom, Bottom>(obj, "Bottom", "Bottom");

   CheckForWithClassInfo<Mid1, Top>(obj, "Mid1", "Top");
   CheckForWithClassInfo<Mid1, Mid1>(obj, "Mid1", "Mid1");
   // Classes are unrelated so should return -1.
   cerr << "The derived class does not derive from base, thus we expect "
      "different results from compiler and cling.\n";
   CheckForWithClassInfo<Mid1, Mid2>(obj, "Mid1", "Mid2");

   //to Bottom or to Mid2, did we not already check for Bottom?
   // Need to cast to Bottom or ambiguous:
   //  struct Basement -> struct Mid2
   //  struct Basement -> struct Bottom -> struct Mid2
   CheckForWithClassInfo<Mid2, Top>((Bottom*)obj, "Mid2", "Top");
   CheckForWithClassInfo<Mid2, Mid2>((Bottom*)obj, "Mid2", "Mid2");
}
