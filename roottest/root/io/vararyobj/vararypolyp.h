// vararypolyp.h
#ifndef VARARYPOLYP_H
#define VARARYPOLYP_H

#include "TObject.h"

class A : public TObject {
public:
   int n;
public:
   A();
   A(const A&);
   ~A() override;
   A& operator=(const A&);
   virtual void clear();
   virtual void set();
   virtual void print();
ClassDefOverride(A, 2)
};

class B : public A {
public:
   int n;
public:
   B();
   B(const B&);
   ~B() override;
   B& operator=(const B&);
   void clear() override;
   void set() override;
   void print() override;
ClassDefOverride(B, 2);
};

class C : public TObject {
public:
   int x;
   A** y; //[x]
   int z;
public:
   C();
   C(const C&);
   ~C() override;
   C& operator=(const C&);
   virtual void clear();
   virtual void set();
   virtual void print();
ClassDefOverride(C, 2);
};

#endif // VARARYPOLYP_H
