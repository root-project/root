// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */


#ifndef   SIGC_OBJECT_SLOT
#define   SIGC_OBJECT_SLOT
#include <sigc++/slot.h>
#include <sigc++/object.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API ObjectSlotNode : public SlotNode
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
    typedef void (Object::*Method)(void);
#endif
    Control_  *control_;
    void      *object_;
    Method     method_;
    Link       link_;
    
    // Can be a dependency 
    virtual Link* link();
    virtual void notify(bool from_child);

    template <class T,class T2>
    ObjectSlotNode(FuncPtr proxy,T* control,void *object,T2 method)
      : SlotNode(proxy)
      { init(control,object,reinterpret_cast<Method&>(method)); }
    void init(Object* control, void* object, Method method);
    virtual ~ObjectSlotNode();
  };



// These do not derive from ObjectSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
template <class R,class Obj>
struct ObjectSlot0_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(void * s) 
      { 
        typedef RType (Obj::*Method)();
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(); 
      }
  };

template <class R,class O1,class O2>
Slot0<R>
  slot(O1& obj,R (O2::*method)())
  { 
    typedef ObjectSlot0_<R,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class O1,class O2>
Slot0<R>
  slot(O1& obj,R (O2::*method)() const)
  {
    typedef ObjectSlot0_<R,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class Obj>
struct ObjectSlot1_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void * s) 
      { 
        typedef RType (Obj::*Method)(P1);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1); 
      }
  };

template <class R,class P1,class O1,class O2>
Slot1<R,P1>
  slot(O1& obj,R (O2::*method)(P1))
  { 
    typedef ObjectSlot1_<R,P1,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class O1,class O2>
Slot1<R,P1>
  slot(O1& obj,R (O2::*method)(P1) const)
  {
    typedef ObjectSlot1_<R,P1,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class P2,class Obj>
struct ObjectSlot2_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2); 
      }
  };

template <class R,class P1,class P2,class O1,class O2>
Slot2<R,P1,P2>
  slot(O1& obj,R (O2::*method)(P1,P2))
  { 
    typedef ObjectSlot2_<R,P1,P2,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class P2,class O1,class O2>
Slot2<R,P1,P2>
  slot(O1& obj,R (O2::*method)(P1,P2) const)
  {
    typedef ObjectSlot2_<R,P1,P2,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class P2,class P3,class Obj>
struct ObjectSlot3_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3); 
      }
  };

template <class R,class P1,class P2,class P3,class O1,class O2>
Slot3<R,P1,P2,P3>
  slot(O1& obj,R (O2::*method)(P1,P2,P3))
  { 
    typedef ObjectSlot3_<R,P1,P2,P3,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class P2,class P3,class O1,class O2>
Slot3<R,P1,P2,P3>
  slot(O1& obj,R (O2::*method)(P1,P2,P3) const)
  {
    typedef ObjectSlot3_<R,P1,P2,P3,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class P2,class P3,class P4,class Obj>
struct ObjectSlot4_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class O1,class O2>
Slot4<R,P1,P2,P3,P4>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4))
  { 
    typedef ObjectSlot4_<R,P1,P2,P3,P4,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class P2,class P3,class P4,class O1,class O2>
Slot4<R,P1,P2,P3,P4>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4) const)
  {
    typedef ObjectSlot4_<R,P1,P2,P3,P4,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class P2,class P3,class P4,class P5,class Obj>
struct ObjectSlot5_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5,class O1,class O2>
Slot5<R,P1,P2,P3,P4,P5>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4,P5))
  { 
    typedef ObjectSlot5_<R,P1,P2,P3,P4,P5,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class P2,class P3,class P4,class P5,class O1,class O2>
Slot5<R,P1,P2,P3,P4,P5>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4,P5) const)
  {
    typedef ObjectSlot5_<R,P1,P2,P3,P4,P5,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }


template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class Obj>
struct ObjectSlot6_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void * s) 
      { 
        typedef RType (Obj::*Method)(P1,P2,P3,P4,P5,P6);
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(p1,p2,p3,p4,p5,p6); 
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class O1,class O2>
Slot6<R,P1,P2,P3,P4,P5,P6>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4,P5,P6))
  { 
    typedef ObjectSlot6_<R,P1,P2,P3,P4,P5,P6,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <class R,class P1,class P2,class P3,class P4,class P5,class P6,class O1,class O2>
Slot6<R,P1,P2,P3,P4,P5,P6>
  slot(O1& obj,R (O2::*method)(P1,P2,P3,P4,P5,P6) const)
  {
    typedef ObjectSlot6_<R,P1,P2,P3,P4,P5,P6,O2> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif /* SIGC_OBJECT_SLOT */

