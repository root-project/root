#ifndef ROOTTEST__templateMembersClasses_h
#define ROOTTEST__templateMembersClasses_h
#include <iostream>
#include <TObject.h>

using namespace std;

#if defined(_MSC_VER) && !defined(__CINT__)
#define SHOW std::cout << __FUNCSIG__ << std::endl;
#define SHOWMEM std::cout << "Mem: "; SHOW
#else
#define SHOW  std::cout << __PRETTY_FUNCTION__ << std::endl
#define SHOWMEM std::cout << "Mem: " << __PRETTY_FUNCTION__ << std::endl
#endif

class Base {
 public:
  Base() { SHOW; }
  Base(const Base&) { SHOWMEM; }
  virtual ~Base() { SHOWMEM; }
  ClassDef(Base,0);
};

class Derived : public Base {
 public:
  Derived() { SHOWMEM; }
  Derived(const Derived& c) : Base(c) { SHOWMEM; }
  virtual ~Derived() { SHOWMEM; }
  ClassDef(Derived,0);
};

template <class T>
class TemplateClass {
 public:
  TemplateClass() { SHOWMEM; }
  TemplateClass(const TemplateClass&) { SHOWMEM; }
  virtual ~TemplateClass() { SHOWMEM; }
  ClassDef(TemplateClass,0);
};

template <class T>
class my_shared_ptr {
 public:
  my_shared_ptr() { SHOWMEM; }
  my_shared_ptr(const my_shared_ptr&) { SHOWMEM; }

  template <class U>
    void f1(U) { SHOW; }

  template <class U>
    void f2() { SHOW; }

  template <class U>
    Derived f3(U) { SHOW; Derived d; return d; }

  template <class U>
    TemplateClass<U> f4(TemplateClass<U>) { SHOW; TemplateClass<U> tn; return tn; }

  template <class U>
    void f5(const my_shared_ptr<U>&) { SHOW; }

  template <class U>
    my_shared_ptr<T> f6(const my_shared_ptr<U>&) { SHOW; return *this; }

  template <class U>
    my_shared_ptr<T> f7(const my_shared_ptr<U>&) { SHOW; return *this; }

  // this works
  typedef my_shared_ptr<T>& reference;
  template <class U>
    reference f8(const my_shared_ptr<U>&)  { SHOW; return *this; }

  // doesn't work.  only difference from f8 is the typedef.
  template <class U>
    my_shared_ptr<T>& n1(const my_shared_ptr<U>&)  { SHOW; return *this; }

  // doesn't work.
  template <class U>
    my_shared_ptr<T>* n2(const my_shared_ptr<U>&)  { SHOW; return this; }

  template <class U>
    my_shared_ptr<T>* /* const */ n3(const my_shared_ptr<U>&)  { SHOW; return this; }

  template <class U>
    my_shared_ptr<T> const*n4(const my_shared_ptr<U>&)  { SHOW; return this; }

  // this works
  //  template <class U>
  //    reference operator=(const my_shared_ptr<U>&) { SHOW; return *this; }

  // only works as above, with return value typedeffed
  template <class U>
      my_shared_ptr<T>& operator=(const my_shared_ptr<U>&) { SHOW; return *this; }

  template <class U>
    my_shared_ptr(const my_shared_ptr<U>&) { SHOW; } //rootcint can't find this

/*   template <class U> */
/*     my_shared_ptr<T>() { SHOW; } //rootcint can't find this */

  virtual ~my_shared_ptr() { SHOWMEM; }

  ClassDef(my_shared_ptr, 0);
};

#endif
