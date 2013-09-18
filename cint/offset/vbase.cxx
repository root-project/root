#include "TInterpreter.h"

struct Fill {
   char S[1024];
};

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

struct Basement: public Fill, public Mid2, public Bottom {
   char FillMore[17];
};

extern "C" int printf(const char*,...);

template <typename DERIVED, typename TARGET>
long CompOffset(const DERIVED* obj) {
   char* addrDerived = (char*) obj;
   char* addrBase = (char*)((const TARGET*)obj);
   // [base1 base2 base3]
   // ^-- obj starts here
   //                   ^-- obj + sizeof(obj)
   //              ^-- offset to base3
   return addrBase - addrDerived;
}

long InterpOffset(const void* obj, const char* derivedClassName, const char* targetClassName) {
   ClassInfo_t* cliDerived = gInterpreter->ClassInfo_Factory(derivedClassName);
   ClassInfo_t* cliTarget = gInterpreter->ClassInfo_Factory(targetClassName);
   long offset = -1;
   // IMPLEMENT:
   // offset = gInterpreter->ClassInfo_GetBaseOffset(cliBasement, cliTarget, obj);
   gInterpreter->ClassInfo_Delete(cliDerived);
   gInterpreter->ClassInfo_Delete(cliTarget);
   return offset;
}

template <typename DERIVED, typename TARGET>
void CheckFor(const DERIVED* obj,
              const char* derivedClassName, const char* targetClassName) {
   printf("%s: Compiler says %ld, TClass says %ld\n",
          targetClassName,
          CompOffset<DERIVED, TARGET>(obj),
          InterpOffset(obj, derivedClassName, targetClassName));
}

void vbase() {
   Basement *obj = new Basement;

   CheckFor<Basement, Top>(obj, "Basement", "Top");
   CheckFor<Basement, Fill>(obj, "Basement", "Fill");
   CheckFor<Basement, Mid1>(obj, "Basement", "Mid1");
   // Ambiguous:
   //   struct Basement -> struct Mid2
   //   struct Basement -> struct Bottom -> struct Mid2
   // Also see check for error at the end.
   // CheckFor<Basement, Mid2>(obj, "Basement", "Mid2");
   CheckFor<Basement, Bottom>(obj, "Basement", "Bottom");
   // Basement doesn't derive from Basement, but this should still be handled (offset 0)
   CheckFor<Basement, Basement>(obj, "Basement", "Basement");

   CheckFor<Bottom, Top>(obj, "Bottom", "Top");
   CheckFor<Bottom, Mid1>(obj, "Bottom", "Mid1");
   CheckFor<Bottom, Mid2>(obj, "Bottom", "Mid2");
   CheckFor<Bottom, Bottom>(obj, "Bottom", "Bottom");

   CheckFor<Mid1, Top>(obj, "Mid1", "Top");
   CheckFor<Mid1, Mid1>(obj, "Mid1", "Mid1");

   // Need to cast to Bottom or ambiguous:
   //  struct Basement -> struct Mid2
   //  struct Basement -> struct Bottom -> struct Mid2
   CheckFor<Mid2, Top>((Bottom*)obj, "Mid2", "Top");
   CheckFor<Mid2, Mid2>((Bottom*)obj, "Mid2", "Mid2");

   // Assert that this cannot be determined, i.e. this function
   // *must* complain about the ambiguous cast:
   //  struct Basement -> struct Mid2
   //  struct Basement -> struct Bottom -> struct Mid2
   InterpOffset(obj, "Basement", "Mid2");
}
