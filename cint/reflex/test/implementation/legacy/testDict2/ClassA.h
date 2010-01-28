#ifndef DICT2_CLASSA_H
#define DICT2_CLASSA_H

#include "ClassM.h"

static int s_i = 97;

class ClassA: public ClassM {
public:
   template <class T> ClassA&
   operator =(const T&) { return *this; }

   ClassA(): fA(s_i) {}            // This is the "constructor"

   virtual ~ClassA() {}           // This is the 'destructor'

   operator int() { return 1; }   // This is operator int()
   int
   a() { return fA; }            // This is a comment of method a()

   void
   // This the previous line
   setA(int v) { // This is a comment of method setA()
      fA = v;
      ClassA aa;
      int i = 3;
      aa = i;
   }            // End of body


   void dummy(int,
              float);    // Dummy comment

private:
   int fA; // This is a comment of fA
};

void
ClassA::dummy(int,
              float) {}

#include <vector>

namespace Bla {
struct Base {
   typedef std::vector<int> MyVector;
   typedef int MyInteger;
   typedef MyInteger MyOtherInteger;
   enum { A, B, C };
   enum { X };
   enum { Y };
   enum { Z };
   enum {};
   enum ENUM2 { D, E, F };
   enum ENUM3_ { G, H, I };
   typedef enum { J, K, L } enum4;
   int i;
   Base(): i(99) {}

   virtual ~Base() {}

protected:
   enum protectedEnum { PA, PB, PC };

private:
   enum privateEnum { QA, QB, QC };

};
struct Left: virtual Base {};
struct Right: virtual Base {};
struct Diamond: Left,
                Right {};

}

#include <iostream>
#include <vector>
#include <utility>

namespace zot {
struct foo_base {
public:
   foo_base(): fBar(4711) {}

   ~foo_base() {}

protected:
   int fBar;
};

class foo: virtual public foo_base {
public:
   int bar();
   void set_bar(int i);
   void set_bar(float f);
   void operator ++();
};

inline int
foo::bar() { return fBar; }

inline void
foo::set_bar(float f) { fBar = int (f); }

inline void
foo::set_bar(int i) { fBar = i; }

inline void
foo::operator ++() { ++fBar; }

} // namespace zot

template <int i> class classVec {
private:
   double arr[i];
};

template class classVec<5>;

#endif // DICT2_CLASSA_H
