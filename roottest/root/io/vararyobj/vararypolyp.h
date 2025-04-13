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
   virtual ~A();
   A& operator=(const A&);
   virtual void clear();
   virtual void set();
   virtual void print();
ClassDef(A, 2)
};

class B : public A {
public:
   int n;
public:
   B();
   B(const B&);
   virtual ~B();
   B& operator=(const B&);
   virtual void clear();
   virtual void set();
   virtual void print();
ClassDef(B, 2);
};

class C : public TObject {
public:
   int x;
   A** y; //[x]
   int z;
public:
   C();
   C(const C&);
   virtual ~C();
   C& operator=(const C&);
   virtual void clear();
   virtual void set();
   virtual void print();
ClassDef(C, 2);
};

#endif // VARARYPOLYP_H
