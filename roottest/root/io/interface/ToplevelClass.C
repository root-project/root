#ifndef ToplevelClass_C
#define ToplevelClass_C

class MyClass {
public:
   int a;
   MyClass() : a(-1) {}
   MyClass(int ia) : a(ia) {}
   ~MyClass() {};
};

#endif
