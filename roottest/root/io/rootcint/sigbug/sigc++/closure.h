// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */


#ifndef   SIGC_CLOSURE
#define   SIGC_CLOSURE
#include <sigc++/slot.h>
#include <sigc++/object.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

class Object;
/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
template <class T_object>
struct Closure_ : public SlotNode
  {
    typedef void (Object::*Method)(void);
    T_object  object_;
    Method     method_;
    
    template <class T,class T2>
    Closure_(FuncPtr proxy,const T_object &object,T2 method)
      : SlotNode(proxy), object_(object), 
        method(reinterpret_cast<Closure_::Method>(method))
      {}
    virtual ~Closure_()
      {}
  };



// These do not derive from Closure, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
template <class R,class O1,class O2>
struct Closure0_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(void * s) 
      { 
        typedef RType (Obj::*Method)();
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(); 
      }
  };

template <class R,class O1,class O2>
Slot0<R>
  closure(const O1& obj,R (O2::*method)())
  { 
    typedef Closure0_<R,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class O1,class O2>
Slot0<R>
  closure(const O1& obj,R (O2::*method)() const)
  {
    typedef Closure0_<R,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }


template <class R,class O1,class O2,class P1>
struct Closure1_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void * s) 
      { 
        typedef RType (Obj::*Method)(P1);
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(p1); 
      }
  };

template <class R,class P1,class O1,class O2>
Slot1<R,P1>
  closure(const O1& obj,R (O2::*method)(P1))
  { 
    typedef Closure1_<R,P1,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class P1,class O1,class O2>
Slot1<R,P1>
  closure(const O1& obj,R (O2::*method)(P1) const)
  {
    typedef Closure1_<R,P1,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }


template <class R,class O1,class O2,class P1,class P2>
struct Closure2_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2);
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(p1,p2); 
      }
  };

template <class R,class P1,class P2,class O1,class O2>
Slot2<R,P1,P2>
  closure(const O1& obj,R (O2::*method)(P1,P2))
  { 
    typedef Closure2_<R,P1,P2,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class P1,class P2,class O1,class O2>
Slot2<R,P1,P2>
  closure(const O1& obj,R (O2::*method)(P1,P2) const)
  {
    typedef Closure2_<R,P1,P2,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }


template <class R,class O1,class O2,class P1,class P2,class P3>
struct Closure3_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3);
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(p1,p2,p3); 
      }
  };

template <class R,class P1,class P2,class P3,class O1,class O2>
Slot3<R,P1,P2,P3>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3))
  { 
    typedef Closure3_<R,P1,P2,P3,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class P1,class P2,class P3,class O1,class O2>
Slot3<R,P1,P2,P3>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3) const)
  {
    typedef Closure3_<R,P1,P2,P3,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }


template <class R,class O1,class O2,class P1,class P2,class P3,class P4>
struct Closure4_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4);
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(p1,p2,p3,p4); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class O1,class O2>
Slot4<R,P1,P2,P3,P4>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3,P4))
  { 
    typedef Closure4_<R,P1,P2,P3,P4,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class P1,class P2,class P3,class P4,class O1,class O2>
Slot4<R,P1,P2,P3,P4>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3,P4) const)
  {
    typedef Closure4_<R,P1,P2,P3,P4,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }


template <class R,class O1,class O2,class P1,class P2,class P3,class P4,class P5>
struct Closure5_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5);
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(p1,p2,p3,p4,p5); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5,class O1,class O2>
Slot5<R,P1,P2,P3,P4,P5>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3,P4,P5))
  { 
    typedef Closure5_<R,P1,P2,P3,P4,P5,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <class R,class P1,class P2,class P3,class P4,class P5,class O1,class O2>
Slot5<R,P1,P2,P3,P4,P5>
  closure(const O1& obj,R (O2::*method)(P1,P2,P3,P4,P5) const)
  {
    typedef Closure5_<R,P1,P2,P3,P4,P5,O2> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_SLOT
