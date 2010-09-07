#ifndef DICT2_FUNCTIONS_H
#define DICT2_FUNCTIONS_H
#include <iostream>

void
function1() {}

int
function2() { return 998; }

double*
function3(int = 9,
          float = 1.0) { static double a = 1.1; return &a; }

struct A {};

float
operator +(A,
           A) { return 0; }

std::ostream&
operator <<(std::ostream& s,
            const A&) { return s; }

namespace Functions {
void
function1() {}

int
function2() { return 999; }

double*
function3(int = 9,
          float = 1.0) { static double a = 1.0; return &a; }

int
function4(int) { return 11; }

}


class ClassAAA {
public:
   int
   function6(int) { return 0; }

};

typedef int MYINT;
int
function5(MYINT) { return 0; }


struct ClassBBB {
   int
   meth(int) { return 1; }

};

typedef int INTTYPEDEF;

int
trigger(INTTYPEDEF s) {
   return s;
}

struct OP {
   operator double() { return 0.; }
   operator ClassAAA() { return ClassAAA(); }
   operator MYINT() { return 0; }
   struct OPS{};
};

namespace FUNCS {
   class C {
   public:
      C() {}
      ~C() {}
      C(const C&) {}
      C& operator=(C) { return *this; }
      operator ClassAAA() { return ClassAAA(); }
      operator MYINT() { return 0; }
      operator OP::OPS() { return OP::OPS(); }
      void f() {}
   };
   C operator +(const ClassAAA&, const ClassBBB&) { return C(); }
   FUNCS::C operator -(const ClassAAA&, const ClassBBB&) { return C(); }
   ::ClassAAA operator*(const ClassAAA&, const ClassBBB&) { return ClassAAA(); }
}

#endif // DICT2_FUNCTIONS_H
