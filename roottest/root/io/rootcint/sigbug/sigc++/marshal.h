// Copyright 2000, Karl Einar Nelson
#ifndef SIGCXX_MARSHALLER_H
#define SIGCXX_MARSHALLER_H
#include <sigc++/sigcconfig.h>
#include <sigc++/trait.h>

#ifndef SIGC_CXX_INT_CTOR
#include <new>
#endif

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/* 

All classes used to marshal return values should have the following API:

  class SomeMarshal
    {
     // both typedefs must be defined.
     typedef Type1 InType;
     typedef Type2 OutType;

     public:
       // Return final return code.
       OutType value();  

       // Captures return codes and returns TRUE to stop emittion.
       bool marshal(const InType&);

       SomeMarshal();
   };

It is not necessary for the InType to match the OutType.
This is to allow for things like list capturing.

*/

/*******************************************************************
***** Marshal 
*******************************************************************/

// Basic Marshal class.  
template <typename R>
class Marshal
  {
   public:
     typedef R OutType;
     typedef typename Trait<R>::type InType;
   protected:
#ifdef SIGC_CXX_PARTIAL_SPEC
     typedef OutType OutType_;
#else
     typedef InType OutType_;
#endif
     OutType_ value_;
   public:
     OutType_& value() {return value_;}

     static OutType_ default_value() 
#ifdef SIGC_CXX_INT_CTOR
       {return OutType_();}
#else
       {OutType_ r; new (&r) OutType_(); return r;}
#endif

     // This captures return values.  Return TRUE to stop emittion process.
     bool marshal(const InType& newval)
       {
        value_=newval;
        return 0;  // continue emittion process
       };
     Marshal()
#ifdef SIGC_CXX_INT_CTOR
       :value_()
       {}
#else
       {
        new (&value_) OutType_();
       }
#endif
  };

#ifdef SIGC_CXX_SPECIALIZE_REFERENCES
// Basic Marshal class.
template <typename R>
class Marshal<R&>
  {
    public:
     typedef R& OutType;
     typedef R& InType;
     R* value_;
     OutType value() {return *value_;}
     static OutType default_value() {return Default;}
     static R Default;

     // This captures return values.  Return TRUE to stop emittion process.
     bool marshal(InType newval)
       {
        value_=&newval;
        return 0;  // continue emittion process
       };
     Marshal()
       :value_(&Default)
       {}
     ~Marshal()
       {}
  };

template <typename T> T Marshal<T&>::Default;
#endif

#ifdef SIGC_CXX_PARTIAL_SPEC
// dummy marshaller for void type.
template <>
class Marshal<void>
  {
   public:
     typedef void OutType;
     static void default_value() {}
     static void value() {}
   Marshal() 
     {}
   ~Marshal()
     {}
  };
#endif

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif
