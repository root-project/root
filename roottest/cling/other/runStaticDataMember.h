#ifndef STATICDATAMEMBER_H
#define STATICDATAMEMBER_H

class A {
public:
  int x;
public:
   A();
   ~A();
};

class B {
public:
   static A a;
public:
   B();
   ~B();
};

#endif // STATICDATAMEMBER_H

