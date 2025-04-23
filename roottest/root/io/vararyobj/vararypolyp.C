// vararypolyp.C

#include "vararypolyp.h"

#include <iostream>

using namespace std;

ClassImp(A)

A::A()
: TObject()
, n(0)
{
}

A::A(const A& rhs)
: TObject(rhs)
, n(rhs.n)
{
}

A& A::operator=(const A& rhs)
{
   if (this != &rhs) {
      n = rhs.n;
   }
   return *this;
}

A::~A()
{
}

void A::clear()
{
   n = 0;
}

void A::set()
{
   n = 1;
}

void A::print()
{
   cout << "n: " << n << endl;
}

ClassImp(B)

B::B()
: A()
{
}

B::B(const B& rhs)
: A(rhs)
{
}

B& B::operator=(const B& rhs)
{
   if (this != &rhs) {
      A::operator=(rhs);
   }
   return *this;
}

B::~B()
{
}

void B::clear()
{
   n = 0;
}

void B::set()
{
   n = 5;
}

void B::print()
{
   cout << "n: " << n << endl;
}

ClassImp(C)

C::C()
: TObject()
, x(0)
, z(0)
{
   y = new A*[3];
   y[0] = new A();
   y[1] = new B();
   y[2] = new A();
}

C::C(const C& rhs)
: TObject(rhs)
, x(rhs.x)
, z(rhs.z)
{
   y = new A*[3];
   y[0] = new A(*rhs.y[0]);
   y[1] = new B(*dynamic_cast<B*>(rhs.y[1]));
   y[2] = new A(*rhs.y[2]);
}

C& C::operator=(const C& rhs)
{
   if (this != &rhs) {
      delete y[0];
      delete y[1];
      delete y[2];
      y[0] = new A(*rhs.y[0]);
      y[1] = new B(*dynamic_cast<B*>(rhs.y[1]));
      y[2] = new A(*rhs.y[2]);
   }
   return *this;
}

C::~C()
{
   delete y[0];
   y[0] = 0;
   delete y[1];
   y[1] = 0;
   delete y[2];
   y[2] = 0;
   delete[] y;
   y = 0;
}

void C::clear()
{
   x = 0;
   y[0]->clear();
   y[1]->clear();
   y[2]->clear();
   z = 0;
}

void C::set()
{
   x = 3;
   y[0]->set();
   y[1]->set();
   y[2]->set();
   z = 7;
}

void C::print()
{
   cout << "x: " << x << endl;
   y[0]->print();
   y[1]->print();
   y[2]->print();
   cout << "z: " << z << endl;
}

void vararypolyp()
{
}

