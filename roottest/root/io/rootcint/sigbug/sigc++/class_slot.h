// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */


#ifndef   SIGC_CLASS_SLOT
#define   SIGC_CLASS_SLOT
#include <sigc++/slot.h>

/*
  SigC::slot_class() (class)
  -----------------------
  slot_class() can be applied to a class method to form a Slot with a
  profile equivalent to the method.  At the same time an instance
  of that class must be specified.  This is an unsafe interface.

  This does NOT require that the class be derived from SigC::Object.
  However, the object should be static with regards to the signal system.
  (allocated within the global scope.)  If it is not and a connected
  slot is call it will result in a segfault.  If the object must
  be destroyed before the connected slots, all connections must
  be disconnected by hand.

  Sample usage:

    struct A
      {
       void foo(int, int);
      } a;

    Slot2<void,int,int> s = slot_class(a, &A::foo);

*/


#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif


/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API ClassSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void (GenericObject::*Method)(void);
public:
#else
    typedef void (SlotNode::*Method)(void);
#endif
    void    *object_;
    Method   method_;
 
    template <class T1, class T2>
    ClassSlotNode(FuncPtr proxy,T1* obj,T2 method)
        : SlotNode(proxy), object_(obj), method_(reinterpret_cast<Method&>(method))
        {}

    virtual ~ClassSlotNode();
  };



// These do not derive from ClassSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
template <class R,class Obj>
struct ClassSlot0_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(void *s) 
      { 
        typedef RType (Obj::*Method)();
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(); 
      }
  };

template <class R,class Obj>
  Slot0<R> 
    slot_class(Obj& obj,R (Obj::*method)())
  { 
    typedef ClassSlot0_<R,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class Obj>
  Slot0<R>
    slot_class(Obj& obj, R (Obj::*method)() const)
  {
    typedef ClassSlot0_<R,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class Obj>
struct ClassSlot1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void *s) 
      { 
        typedef RType (Obj::*Method)(P1);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1); 
      }
  };

template <class R,class P1,class Obj>
  Slot1<R,P1> 
    slot_class(Obj& obj,R (Obj::*method)(P1))
  { 
    typedef ClassSlot1_<R,P1,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class Obj>
  Slot1<R,P1>
    slot_class(Obj& obj, R (Obj::*method)(P1) const)
  {
    typedef ClassSlot1_<R,P1,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class P2,class Obj>
struct ClassSlot2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *s) 
      { 
        typedef RType (Obj::*Method)(P1,P2);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2); 
      }
  };

template <class R,class P1,class P2,class Obj>
  Slot2<R,P1,P2> 
    slot_class(Obj& obj,R (Obj::*method)(P1,P2))
  { 
    typedef ClassSlot2_<R,P1,P2,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class P2,class Obj>
  Slot2<R,P1,P2>
    slot_class(Obj& obj, R (Obj::*method)(P1,P2) const)
  {
    typedef ClassSlot2_<R,P1,P2,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class P2,class P3,class Obj>
struct ClassSlot3_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3); 
      }
  };

template <class R,class P1,class P2,class P3,class Obj>
  Slot3<R,P1,P2,P3> 
    slot_class(Obj& obj,R (Obj::*method)(P1,P2,P3))
  { 
    typedef ClassSlot3_<R,P1,P2,P3,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class P2,class P3,class Obj>
  Slot3<R,P1,P2,P3>
    slot_class(Obj& obj, R (Obj::*method)(P1,P2,P3) const)
  {
    typedef ClassSlot3_<R,P1,P2,P3,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class P2,class P3,class P4,class Obj>
struct ClassSlot4_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class Obj>
  Slot4<R,P1,P2,P3,P4> 
    slot_class(Obj& obj,R (Obj::*method)(P1,P2,P3,P4))
  { 
    typedef ClassSlot4_<R,P1,P2,P3,P4,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class P2,class P3,class P4,class Obj>
  Slot4<R,P1,P2,P3,P4>
    slot_class(Obj& obj, R (Obj::*method)(P1,P2,P3,P4) const)
  {
    typedef ClassSlot4_<R,P1,P2,P3,P4,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class P2,class P3,class P4,class P5,class Obj>
struct ClassSlot5_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5,class Obj>
  Slot5<R,P1,P2,P3,P4,P5> 
    slot_class(Obj& obj,R (Obj::*method)(P1,P2,P3,P4,P5))
  { 
    typedef ClassSlot5_<R,P1,P2,P3,P4,P5,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class P2,class P3,class P4,class P5,class Obj>
  Slot5<R,P1,P2,P3,P4,P5>
    slot_class(Obj& obj, R (Obj::*method)(P1,P2,P3,P4,P5) const)
  {
    typedef ClassSlot5_<R,P1,P2,P3,P4,P5,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }


template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class Obj>
struct ClassSlot6_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void *s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5,P6);
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5,p6); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class Obj>
  Slot6<R,P1,P2,P3,P4,P5,P6> 
    slot_class(Obj& obj,R (Obj::*method)(P1,P2,P3,P4,P5,P6))
  { 
    typedef ClassSlot6_<R,P1,P2,P3,P4,P5,P6,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class Obj>
  Slot6<R,P1,P2,P3,P4,P5,P6>
    slot_class(Obj& obj, R (Obj::*method)(P1,P2,P3,P4,P5,P6) const)
  {
    typedef ClassSlot6_<R,P1,P2,P3,P4,P5,P6,Obj> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif /* SIGC_CLASS_SLOT */

