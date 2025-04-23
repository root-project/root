// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */


#ifndef   SIGC_METHOD_SLOT
#define   SIGC_METHOD_SLOT
#include <sigc++/slot.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/**************************************************************/
// These are internal classes used to represent function varients of slots

class Object;
// (internal) 
struct LIBSIGC_API MethodSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void* (GenericObject::*Method)(void*);
public:
#else
    typedef void* (Object::*Method)(void*);
#endif
    Method     method_;
    
    template <class T2>
    MethodSlotNode(FuncPtr proxy,T2 method)
      : SlotNode(proxy)
      { init(reinterpret_cast<Method&>(method)); }
    void init(Method method);
    virtual ~MethodSlotNode();
  };

struct LIBSIGC_API ConstMethodSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void* (GenericObject::*Method)(void*) const;
public:
#else
    typedef void* (Object::*Method)(void*) const;
#endif
    Method     method_;
    
    template <class T2>
    ConstMethodSlotNode(FuncPtr proxy,T2 method)
      : SlotNode(proxy)
      { init(reinterpret_cast<Method&>(method)); }
    void init(Method method);
    virtual ~ConstMethodSlotNode();
  };



// These do not derive from MethodSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
template <class R,class Obj>
struct MethodSlot0_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,void * s) 
      { 
        typedef RType (Obj::*Method)();
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))();
      }
  };

template <class R,class Obj>
struct ConstMethodSlot0_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,void * s) 
      { 
        typedef RType (Obj::*Method)() const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))();
      }
  };

template <class R,class Obj>
Slot1<R,Obj&>
  slot(R (Obj::*method)())
  { 
    typedef MethodSlot0_<R,Obj> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj>
Slot1<R,const Obj&>
  slot(R (Obj::*method)() const)
  {
    typedef ConstMethodSlot0_<R,Obj> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }


template <class R,class Obj,class P1>
struct MethodSlot1_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,void * s) 
      { 
        typedef RType (Obj::*Method)(P1);
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1);
      }
  };

template <class R,class Obj,class P1>
struct ConstMethodSlot1_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,void * s) 
      { 
        typedef RType (Obj::*Method)(P1) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1);
      }
  };

template <class R,class Obj,class P1>
Slot2<R,Obj&,P1>
  slot(R (Obj::*method)(P1))
  { 
    typedef MethodSlot1_<R,Obj,P1> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj,class P1>
Slot2<R,const Obj&,P1>
  slot(R (Obj::*method)(P1) const)
  {
    typedef ConstMethodSlot1_<R,Obj,P1> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }


template <class R,class Obj,class P1,class P2>
struct MethodSlot2_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2);
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2);
      }
  };

template <class R,class Obj,class P1,class P2>
struct ConstMethodSlot2_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2);
      }
  };

template <class R,class Obj,class P1,class P2>
Slot3<R,Obj&,P1,P2>
  slot(R (Obj::*method)(P1,P2))
  { 
    typedef MethodSlot2_<R,Obj,P1,P2> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj,class P1,class P2>
Slot3<R,const Obj&,P1,P2>
  slot(R (Obj::*method)(P1,P2) const)
  {
    typedef ConstMethodSlot2_<R,Obj,P1,P2> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }


template <class R,class Obj,class P1,class P2,class P3>
struct MethodSlot3_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3);
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3);
      }
  };

template <class R,class Obj,class P1,class P2,class P3>
struct ConstMethodSlot3_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3);
      }
  };

template <class R,class Obj,class P1,class P2,class P3>
Slot4<R,Obj&,P1,P2,P3>
  slot(R (Obj::*method)(P1,P2,P3))
  { 
    typedef MethodSlot3_<R,Obj,P1,P2,P3> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj,class P1,class P2,class P3>
Slot4<R,const Obj&,P1,P2,P3>
  slot(R (Obj::*method)(P1,P2,P3) const)
  {
    typedef ConstMethodSlot3_<R,Obj,P1,P2,P3> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }


template <class R,class Obj,class P1,class P2,class P3,class P4>
struct MethodSlot4_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4);
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4);
      }
  };

template <class R,class Obj,class P1,class P2,class P3,class P4>
struct ConstMethodSlot4_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4);
      }
  };

template <class R,class Obj,class P1,class P2,class P3,class P4>
Slot5<R,Obj&,P1,P2,P3,P4>
  slot(R (Obj::*method)(P1,P2,P3,P4))
  { 
    typedef MethodSlot4_<R,Obj,P1,P2,P3,P4> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj,class P1,class P2,class P3,class P4>
Slot5<R,const Obj&,P1,P2,P3,P4>
  slot(R (Obj::*method)(P1,P2,P3,P4) const)
  {
    typedef ConstMethodSlot4_<R,Obj,P1,P2,P3,P4> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }


template <class R,class Obj,class P1,class P2,class P3,class P4,class P5>
struct MethodSlot5_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5);
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5);
      }
  };

template <class R,class Obj,class P1,class P2,class P3,class P4,class P5>
struct ConstMethodSlot5_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(Obj& obj,typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5);
      }
  };

template <class R,class Obj,class P1,class P2,class P3,class P4,class P5>
Slot6<R,Obj&,P1,P2,P3,P4,P5>
  slot(R (Obj::*method)(P1,P2,P3,P4,P5))
  { 
    typedef MethodSlot5_<R,Obj,P1,P2,P3,P4,P5> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <class R,class Obj,class P1,class P2,class P3,class P4,class P5>
Slot6<R,const Obj&,P1,P2,P3,P4,P5>
  slot(R (Obj::*method)(P1,P2,P3,P4,P5) const)
  {
    typedef ConstMethodSlot5_<R,Obj,P1,P2,P3,P4,P5> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_SLOT
