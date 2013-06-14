//
//  with ACLiC run as "root -b -q templatefriend.cxx++"
//  with gcc on Linux/Macosx as "gcc -o templatefriend templatefriend.cxx -lstdc++"
//
#include <iostream>
#include <typeinfo>

// MSVC2010 pulls in std::tr1::shared_ptr, C++11 has std::shared_ptr
// and Cling pulls in std,
// which combined creates ambiguities.
# define shared_ptr not_tr1_shared_ptr

class Parent{ public: static const char *ClassName() { return "Parent"; } };

class Child : public Parent { public: static const char *ClassName() { return "Child"; }};

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
   void PF(const char* func) {
      std::string clname;
#if defined(__CINT__) || defined(_MSC_VER)
      clname = T::ClassName();
#else
      clname = __PRETTY_FUNCTION__;
      if (clname.find("Parent") != std::string::npos) clname = "Parent";
      else clname = "Child";
#endif
      std::cout << "shared_ptr<" << clname << ">::" << func << std::endl;
   }

public:

  template <class U> friend class shared_ptr;

  const char *GetName() { return typeid(T).name(); }

  shared_ptr(T* someobject) 
  { 
     PF("c'tor(T)");
    theobject = someobject; 
  }

  template <class Y>
  shared_ptr(shared_ptr<Y> const &rhs) 
    : theobject(dynamic_cast<T*>(rhs.theobject)) 
  {
     PF("c'tor(Y)");
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

