#include <iostream>
#include <TObject.h>

using namespace std;

#define SHOW std::cout << __PRETTY_FUNCTION__ << std::endl

class Base {
 public:
  Base() { SHOW; }
  virtual ~Base() { SHOW; }
  ClassDef(Base,0);
};

class Derived : public Base {
 public:
  Derived() { SHOW; }
  virtual ~Derived() { SHOW; }
  ClassDef(Derived,0);
};

template <class T>
class TemplateClass {
 public:
  TemplateClass() { SHOW; }
  virtual ~TemplateClass() { SHOW; }
  ClassDef(TemplateClass,0);
};

template <class T>
class shared_ptr {
 public:
  shared_ptr() { SHOW; }

  template <class U>
    void f1(U) { SHOW; }

  template <class U>
    void f2() { SHOW; }

  template <class U>
    Derived f3(U) { SHOW; Derived d; return d; }

  template <class U>
    TemplateClass<U> f4(TemplateClass<U>) { SHOW; TemplateClass<U> tn; return tn; }

  template <class U>
    void f5(const shared_ptr<U>&) { SHOW; }

  template <class U>
    shared_ptr<T> f6(const shared_ptr<U>&) { SHOW; return *this; }

  template <class U>
    shared_ptr<T> f7(const shared_ptr<U>&) { SHOW; return *this; }

  // this works
  typedef shared_ptr<T>& reference;
  template <class U>
    reference f8(const shared_ptr<U>&)  { SHOW; return *this; }

  // doesn't work.  only difference from f8 is the typedef.
  template <class U>
    shared_ptr<T>& n1(const shared_ptr<U>&)  { SHOW; return *this; }

  // doesn't work. 
  template <class U>
    shared_ptr<T>* n2(const shared_ptr<U>&)  { SHOW; return this; }

  template <class U>
    shared_ptr<T>* const n3(const shared_ptr<U>&)  { SHOW; return this; }

  template <class U>
    shared_ptr<T> const*n4(const shared_ptr<U>&)  { SHOW; return this; }

  // this works
  //  template <class U>
  //    reference operator=(const shared_ptr<U>&) { SHOW; return *this; } 

  // only works as above, with return value typedeffed
  template <class U>
      shared_ptr<T>& operator=(const shared_ptr<U>&) { SHOW; return *this; } 

  template <class U>
    shared_ptr<T>(const shared_ptr<U>&) { SHOW; } //rootcint can't find this

/*   template <class U> */
/*     shared_ptr<T>() { SHOW; } //rootcint can't find this */

  virtual ~shared_ptr() { SHOW; }

  ClassDef(shared_ptr, 0);
};


