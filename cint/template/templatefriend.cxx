//
//  with ACLiC run as "root -b -q templatefriend.cxx++"
//  with gcc on Linux/Macosx as "gcc -o templatefriend templatefriend.cxx -lstdc++"
//
#include <iostream>
#include <typeinfo>

class Parent{ };

class Child : public Parent { };

template <class T> class shared_ptr;

class testing {
   template <class U> friend class shared_ptr;
};

template <class T>
class shared_ptr 
{

  T *theobject;
  // this shared_ptr class is of course missing any 
  // reference counting mechanism

public:

  template <class U> friend class shared_ptr;

  const char *GetName() { return typeid(T).name(); }

  shared_ptr(T* someobject) 
  { 
#ifdef __CINT__
    std::cout << "shared_ptr<T>::shared_ptr(T *) [with T = " << typeid(T).name() << "]" << endl;
#else
    std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
    theobject = someobject; 
  }

  template <class Y>
  shared_ptr(shared_ptr<Y> const &rhs) 
    : theobject(dynamic_cast<T*>(rhs.theobject)) 
  {
#ifdef __CINT__
     std::cout << "shared_ptr<T>::shared_ptr(const shared_ptr<Y> &) [with Y = " << typeid(Y).name()
               << ", T = " << typeid(T).name() << "]" << endl;
#else
    std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
  };
};

#ifdef __MAKECINT__
#pragma link C++ class shared_ptr<Parent>+;
#pragma link C++ class shared_ptr<Child>+;
#endif
int templatefriend()
{
  // create shared pointer to child
  shared_ptr<Child> child_p(new Child); 

  // create shared pointer to parent from 
  // shared pointer to child, calls dynamic_cast<> in template constructor.
  shared_ptr<Parent> parent_p(child_p); 
  return 0;
}

#ifndef __MAKECINT__
int main() { return templatefriend(); }
#endif

