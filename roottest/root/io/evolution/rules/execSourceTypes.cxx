#include "TClass.h"
#include "TFile.h"
#include "TList.h"
#include "TObjString.h"
#include "TStreamerInfo.h"

#include <limits>
#include <type_traits>
#include <list>
#include <string>

struct A {
  float f = 1.1;
};

struct B {
  double f = 2.2;
};

struct C {
  long long f = 5;
};

struct RooLikeList : public TList {
   ClassDef(RooLikeList, 2);
};

void Feed(TList &l, std::vector<const char*> names)
{
   for(auto name : names)
      l.Add(new TObjString(name));
}

struct Old
{
   // When this was missing from the 'Old' (i.e. comment the next line)
   // and used in a read rule, the offset for the cached arrays was wrong - set to 0
   float fSingle = 1.0;
   int fInt = 1;
   int fHitPattern[3] = {101, 102, 103};
   unsigned int fHitCount = 2;
   A fValue{1.1};
   A fValueB{1.2};
   A fValueC{1.3};;
   A *fPtr = new A{10.1};
   A *fPtrB = new A{10.2};
   A *fPtrC = new A{10.3};
   A fArray[3] = {1.5, 2.6, 3.7};
   A fArrayB[3] = {11.5, 12.6, 13.7};
   A fArrayC[3] = {21.5, 22.6, 23.7};
   A fArraySkip[3] = {31.5, 32.6, 33.7};
   A fArrayS[3] = {41.5, 42.6, 43.7};
   A fArraySB[3] = {51.5, 52.6, 53.7};
   A fArraySC[3] = {61.5, 62.6, 63.7};
   RooLikeList fListImplicit;
   RooLikeList fListExplicit;
   RooLikeList fListImplicitExplicit;

   std::string fString = "input message";

   Old() {
      Feed(fListImplicit, {"a1", "a2", "a3"});
      Feed(fListExplicit, {"b1", "b2", "b3"});
      Feed(fListImplicitExplicit, {"c1", "c2", "c3"});
   }

   ~Old()
   {
      delete fPtr;
      delete fPtrB;
      delete fPtrC;
      fListImplicit.Delete();
      fListExplicit.Delete();
      fListImplicitExplicit.Delete();
   }
};

struct New
{
   double fInt = 2.0;
   double fSingle = 10.0;
   double fExtra  = 15.0;
   double fDouble = 20.0;
   int fHitPattern[3] = {101, 102, 103};
   int fHitCount = 2;
   B fValue;  // Using implicit conversion
   B fValueB; // Using explicit conversion from A to B
   C fValueC; // Using implicit conversion from A to B and explicit conversion to C
   B *fPtr = nullptr;  // Using implicit conversion
   B *fPtrB = nullptr; // Read the A* into a A* and assign its 'value' to a B*
   C *fPtrC = nullptr; // Read the A* into a B* and assign its 'value' to a C*

   B fArray[3];  // Using implicit conversion
   B fArrayB[3]; // Using explicit conversion from A to B
   C fArrayC[3]; // Using implicit conversion from A to B and explicit conversion to C

   // We are intentionally not adding a member named:
   //   fArraySkip
   // to test skipping array of objects

   std::array<B, 3> fArrayS;  // Using implicit conversion
   std::array<B, 3> fArraySB; // Using explicit conversion from A[] to std::array of B
   std::array<C, 3> fArraySC;  // Using implicit conversion from A to std::array of B and explicit conversion to C

   // This member has intentionally no correspoinding member in the input
   C fArrayNew[3];

   TList fListImplicit;
   TList fListExplicit;
   std::list<std::string> fListImplicitExplicit;

   std::string fString;

   ~New()
   {
      delete fPtr;
      delete fPtrB;
      delete fPtrC;
      fListImplicit.Delete();
      fListExplicit.Delete();
   }

};


#include <iostream>

void examine(int a, bool trailing = true)
{
   std::cout << "The int value is: " << a;
   if (trailing)
      std::cout << '\n';
}

void examine(double a, bool trailing = true)
{
   std::cout << "The double value is: " << a;
   if (trailing)
      std::cout << '\n';
}

void examine(const std::string &a, bool trailing = true)
{
   std::cout << "The string value is: \"" << a << "\"";
   if (trailing)
      std::cout << '\n';
}

template<typename T>
void examine(const T &a, bool trailing = true)
{
   std::cout << "The " << TClass::GetClass(typeid(T))->GetName();
   std::cout << " value is: " << a.f;
   if (trailing)
      std::cout << '\n';
}

template <typename T>
void examine(T * const p, bool trailing = true)
{
   std::cout << "The " << TClass::GetClass(typeid(T))->GetName() << "* value is: ";
   if (p)
     std::cout << p->f;
   else
     std::cout << "nullptr";
   if (trailing)
     std::cout << '\n';
}

template <int N>
void examine_array(int arr[N], bool trailing = true)
{
   std::cout << "The int[" << N << "] values are:\n";
   std::cout << "   ";
   for(size_t i = 0; i < N; ++i)
      std::cout << arr[i] << "  ";
   if (trailing)
      std::cout << '\n';
}

template <typename T, int N>
void examine_array(T arr[N], bool trailing = true)
{
   std::cout << "The " << TClass::GetClass(typeid(T))->GetName() << "[" << N << "] values are:\n";
   std::cout << "   ";
   for(size_t i = 0; i < N; ++i)
      std::cout << arr[i].f << "  ";
   if (trailing)
      std::cout << '\n';
}

template <typename T, int N>
void examine_array(std::array<T, N> arr, bool trailing = true)
{
   examine_array<T, N>(arr.data(), trailing);
}

void examine_items(const TList &l, bool trailing = true)
{
   for(auto str : TRangeDynCast<TObjString>(l)) {
      std::cout << "   " << str->String();
      if (trailing)
         std::cout << '\n';
      else
         std::cout << "   ";
   }
}

void examine(const TList &l, bool trailing = true)
{
   std::cout << "The TList values are:\n";
   examine_items(l, trailing);
}

void examine(const RooLikeList &l, bool trailing = true)
{
   std::cout << "The RooLikeList values are:\n";
   examine_items(l, trailing);
}

void examine(const std::list<std::string> &l, bool trailing = true)
{
   std::cout << "The std::list<std::string> values are:\n";
   for(auto str : l) {
      std::cout << "   " << str;
      if (trailing)
         std::cout << '\n';
      else
         std::cout << "   ";
   }
}

bool test_value(int in, int ref)
{
   bool error = ( in != ref);
   if (error)
     std::cout << " and is incorrect, expected value: " << ref << '\n';
   else
     std::cout << " and is correct.\n";
   return !error;
}

bool test_value(const std::string &in, const char *ref)
{
   bool error = (in != ref);
   if (error)
     std::cout << " and is incorrect, expected value: \"" << ref << "\"\n";
   else
     std::cout << " and is correct.\n";
   return !error;
}

template<typename T, typename V>
bool test_value(T in, V ref)
{
   bool error = ( std::abs(in-ref) > std::numeric_limits<V>::epsilon() );
   if (error)
     std::cout << " and is incorrect, expected value: " << ref << '\n';
   else
     std::cout << " and is correct.\n";
   return !error;
}

bool check(int in, int ref)
{
   examine(in, false);
   return test_value(in, ref);
}

template<typename V>
bool check(double in, V ref)
{
   examine(in, false);
   return test_value(in, ref);
}

bool check(std::string &in, const char *ref)
{
   examine(in, false);
   return test_value(in, ref);
}

template<typename T, typename V = decltype(T::f)>
bool check(T &a, V ref)
{
   examine(a, false);
   return test_value(a.f, ref);
}

template<typename T, typename V = decltype(T::f)>
bool check(T *p, V ref)
{
   bool error = (p == nullptr) || ( std::abs(p->f-ref) > std::numeric_limits<V>::epsilon() );
   examine(p, false);
   if (error)
     std::cout << " and is incorrect, expected value: " << ref << '\n';
   else
     std::cout << " and is correct.\n";
   return !error;
}

template<size_t N, typename V> //  = decltype(T::f)[N]>
bool check_array(int arr[N], V ref)
{
   bool error = false;
   for(size_t i = 0; i < N; ++i) {
     error = error || ( arr[i] != ref[i]);
   }

   examine_array<N>(arr, false);
   if (error) {
     std::cout << " and is incorrect, expected value: ";
     for(const auto &v : ref)
        std::cout << v << " ";
     std::cout << '\n';
   } else
     std::cout << " and is correct.\n";
   return !error;
}

template<typename T, size_t N, typename V> //  = decltype(T::f)[N]>
bool check_array(T arr[N], V ref)
{
   bool error = false;
   for(size_t i = 0; i < N; ++i) {
     error = error || ( std::abs(arr[i].f - ref[i]) > std::numeric_limits<typename V::value_type>::epsilon() );
   }

   examine_array<T,N>(arr, false);
   if (error) {
     std::cout << " and is incorrect, expected value: ";
     for(const auto &v : ref)
        std::cout << v << " ";
     std::cout << '\n';
   } else
     std::cout << " and is correct.\n";
   return !error;
}

template<typename T, size_t N, typename V> //  = decltype(T::f)[N]>
bool check_array(std::array<T, N> &arr, V ref)
{
   return check_array<T, N, V>(arr.data(), ref);
}

bool check(const TList &l, std::vector<const char*> ref)
{
   bool error = false;
   size_t i = 0;
   for(auto str : TRangeDynCast<TObjString>(l)) {
      error = error || (strcmp(str->String(), ref[i]) != 0);
      ++i;
   }
   if (i != ref.size())
      error = true;

   examine(l);
   if (error) {
      std::cout << " and is incorrect, expected values: ";
      for(const auto &v : ref)
         std::cout << v << " ";
      std::cout << '\n';
   } else
      std::cout << " and is correct.\n";
   return !error;
}

bool check(const std::list<std::string> &l, std::vector<const char*> ref)
{
   bool error = false;
   size_t i = 0;
   for(auto str : l) {
      error = error || (str != ref[i]);
      ++i;
   }
   if (i != ref.size())
      error = true;

   examine(l);
   if (error) {
      std::cout << " and is incorrect, expected values: ";
      for(const auto &v : ref)
         std::cout << v << " ";
      std::cout << '\n';
   } else
      std::cout << " and is correct.\n";
   return !error;
}

void printAddress(void *obj, const char *name, void *member)
{
   std::cout << "For object " << obj << " " << name << " is " << member << '\n';
}

void MoveTo(TList &input, TList &output)
{
   for(auto str : TRangeDynCast<TObjString>(&input))
      output.Add(str);
   input.Clear();
}

void CopyTo(const TList &input, std::list<std::string> &output)
{
   for(auto str : TRangeDynCast<TObjString>(&input))
      output.push_back(str->String().Data());
}

#ifdef __ROOTCLING__
#pragma link C++ class A+;
#pragma link C++ class B+;
#pragma link C++ class C+;
#pragma link C++ class RooLikeList+;
#pragma link C++ class Old+;
#pragma link C++ class New+;
#pragma read sourceClass="A" targetClass="B";
#pragma read sourceClass="A" targetClass="C";
#pragma read sourceClass="B" targetClass="C";
#pragma read sourceClass="Old" targetClass="New";
#pragma read sourceClass="RooLikeList" targetClass="TList";

#pragma read sourceClass="Old" targetClass="New" source="float fInt" target="fInt" version="[1-]" \
   code="{ fInt = onfile.fInt - 1; }"

#pragma read sourceClass="Old" targetClass="New" source="float fSingle" target="fSingle" version="[1-]" \
   code="{ fSingle = onfile.fSingle * 3; }"

#pragma read sourceClass="Old" targetClass="New" source="float fSingle" target="fExtra" version="[1-]" \
   code="{ fExtra = onfile.fSingle * 5; }"

#pragma read sourceClass="Old" targetClass="New" source="double fSingle" target="fDouble" version="[1-]" \
   code="{ std::cout << \"ERROR: the rule for fDouble should not be run\n\"; fDouble = onfile.fSingle * 3; }"

// Objects
#pragma read sourceClass="Old" targetClass="New" source="A fValueB" target="fValueB" version="[1-]" \
   code="{ examine(onfile.fValueB); /* examine(*(B*)&(onfile.fValueB)); */ fValueB.f = onfile.fValueB.f; }"
#pragma read sourceClass="Old" targetClass="New" source="B fValueC" target="fValueC" version="[1-]" \
   code="{ examine(onfile.fValueC); examine(*(B*)&(onfile.fValueC)); fValueC.f = onfile.fValueC.f; }"
// We need a test checking in the case of 2 rules that applies to the same version and has the same input but different type, one or both the rules are rejected.

// Pointers
#pragma read sourceClass="Old" targetClass="New" source="A* fPtrB" target="fPtrB" version="[1-]" \
   code="{ examine(onfile.fPtrB); /* examine((A*)onfile.fPtrB); */ fPtrB = new B{ onfile.fPtrB->f }; }"
// This does not work for 2 reasons at the moment.
// (a) Reading a A* into a B* implicitly does not work (fixed)
// (b) The code parsing the source member type to load into a TStreamerElement does not parse pointer nor arrays.
#pragma read sourceClass="Old" targetClass="New" source="B* fPtrC" target="fPtrC" version="[1-]" \
   code="{ examine(onfile.fPtrC); /* examine((A*)onfile.fPtrC); */ fPtrC = new C{ (long long) onfile.fPtrC->f }; }"

// Arrays
// (fixed) Having a rule with an array input fails, seemingly because the artificial element is not setup correctly (offset 0 in the cached object rather than the actually location of the array)
#pragma read sourceClass="Old" targetClass="New" source="A fArrayB[3]" target="fArrayB" version="[1-]" \
   code="{ examine_array<A,3>(onfile.fArrayB); /* printAddress(newObj, \"fArrayB\", fArrayB); */ for(size_t i = 0; i < 3; ++i) fArrayB[i].f = onfile.fArrayB[i].f; }"

// The implicit conversion for array of objects is broken
#pragma read sourceClass="Old" targetClass="New" source="B fArrayC[3]" target="fArrayC" version="[1-]" \
   code="{ examine_array<B,3>(onfile.fArrayC); for(size_t i = 0; i < 3; ++i) fArrayC[i].f = onfile.fArrayC[i].f; }"


#pragma read sourceClass="Old" targetClass="New" source="std::array<A,3> fArraySB" target="fArraySB" version="[1-]" \
   code="{ examine_array<A,3>(onfile.fArraySB); for(size_t i = 0; i < 3; ++i) fArraySB[i].f = onfile.fArraySB[i].f; }"

#pragma read sourceClass="Old" targetClass="New" source="std::array<B, 3> fArraySC" target="fArraySC" version="[1-]" \
   code="{ examine_array<B,3>(onfile.fArraySC); for(size_t i = 0; i < 3; ++i) fArraySC[i].f = onfile.fArraySC[i].f; }"


#pragma read sourceClass="Old" targetClass="New" source="A fArrayNew[3]" target="fArrayNew" version="[1-]" \
  code="{ examine_array<A,3>(onfile.fArrayNew); for(size_t i = 0; i < 3; ++i) fArrayNew[i].f = onfile.fArrayNew[i].f; }"

// Lists
#pragma read sourceClass="Old" targetClass="New" source="RooLikeList fListExplicit" target="fListExplicit" version="[1-]" \
  code="{ examine(onfile.fListExplicit); MoveTo(onfile.fListExplicit, fListExplicit); }"

#pragma read sourceClass="Old" targetClass="New" source="TList fListImplicitExplicit" target="fListImplicitExplicit" version="[1-]" \
  code="{ examine(onfile.fListImplicitExplicit); CopyTo(onfile.fListImplicitExplicit, fListImplicitExplicit); onfile.fListImplicitExplicit.Delete(); }"

// Rules with 2 (or more) input with the first one being an array.
#pragma read sourceClass="Old" targetClass="New" source="int fHitPattern[3]; int fHitCount" \
  target="fHitPattern,fHitCount" version="[1-]" \
  code="{ for(size_t i = 0; i < 3; ++i) fHitPattern[i] = onfile.fHitPattern[i]+10; fHitCount = onfile.fHitCount + 1; }"

// std::string
#pragma read sourceClass="Old" targetClass="New" source="std::string fString;" target="fString"  version="[1-]" \
  code="{ fString = onfile.fString; }";

#endif

void writefile(const char *filename)
{
   Old o;
   examine_array<A,3>(o.fArray);
   TFile file(filename, "RECREATE");
   file.WriteObject(&o, "oldobject");
   file.Write();
};

int readfile(const char *filename = "sourcetypes.root")
{
   TFile file(filename, "READ");
   if (file.IsZombie())
     return 1;
   auto n = file.Get<New>("oldobject");
   if (!n) {
     std::cerr << "Error: could not find `oldobject`\n";
     return 2;
   }
   if (gDebug > 1) {
      auto info = TClass::GetClass("New")->GetConversionStreamerInfo(TClass::GetClass("Old"), -1);
      info->ls();
      info = TClass::GetClass("Old@@1")->GetStreamerInfo();
      info->ls();
   }

   bool res = check(n->fInt, 0.0);
   res = res && check(n->fSingle, 3.0f);
   res = res && check(n->fExtra, 5.0f);
   res = res && check(n->fDouble, 20.0); // Testing that the rule did *not* run.

   res = res && check(n->fValue, 1.1f);
   res = res && check(n->fValueB, 1.2f);
   res = res && check(n->fValueC, 1);

   // std::cout << "Conversion from A* to B* (data member fBrokenPtr) seems silenty broken without a rule:\n";
   res = res && check(n->fPtr, 10.1f);
   res = res && check(n->fPtrB, 10.2f);
   // fPtrC is known to not be right (it is meant to have a rule using implicit conversion from A* to C* (broken) and then to the rule (also broken)
   res = res && check(n->fPtrC, 10);

   // This is broken/gives odd results
   // The output is "0.225  2.2  2.2" instead of "1.5  1.6  1.7"
   res = res && check_array<B,3>(n->fArray, std::vector<float>{1.5f, 2.6f, 3.7f});
   // This was previous broken due to the offset of the artificial element being wrongly
   // calculated. The output was "2.2  2.2  2.2"
   res = res && check_array<B,3>(n->fArrayB, std::vector<float>{11.5f, 12.6f, 13.7f});
   // Implicit conversion of array of object does not work.
   res = res && check_array<C,3>(n->fArrayC, std::vector<int>{21, 22, 23});

   res = res && check_array(n->fArrayS, std::vector<float>{41.5, 42.6, 43.7});
   res = res && check_array(n->fArraySB, std::vector<float>{51.5, 52.6, 53.7});
   res = res && check_array(n->fArraySC, std::vector<int>{61, 62, 63});

   res = res && check(n->fListImplicit, {"a1", "a2", "a3"});
   res = res && check(n->fListExplicit, {"b1", "b2", "b3"});
   res = res && check(n->fListImplicitExplicit, {"c1", "c2", "c3"});

   res = res && check_array<3>(n->fHitPattern, std::vector<int>{111, 112, 113});
   res = res && check(n->fHitCount, 3);

   res = res && check(n->fString, "input message");

   return !res; // 0 is success.
}

int execSourceTypes()
{
   writefile("sourcestype.root");
   return readfile("sourcestype.root");
}
