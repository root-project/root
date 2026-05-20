// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_RETYPE_H
#define   SIGC_RETYPE_H
#include <sigc++/adaptor.h>

/** @defgroup retype
 * retype() alters a Slot to change arguments and return type.
 *
 * Only allowed conversions or conversions to void can properly
 * be implemented.  The types of the return and all arguments must always
 * be specified as a template parameters.
 *
 * Simple sample usage:
 *
 * @code
 *  float f(float,float);
 *
 *  SigC::Slot1<int, int, int>  s1 = SigC::retype<int, int, int>(slot(&f));
 *  @endcode
 *
 *
 * SigC::retype_return() alters a Slot by changing the return type.
 *
 * Only allowed conversions or conversions to void can properly
 * be implemented.  The type must always be specified as a
 * template parameter.
 *
 * Simple sample usage:
 *
 * @code
 * int f(int);
 *
 * SigC::Slot1<void, int> s1 = SigC::retype_return<void>(slot(&f));
 * SigC::Slot1<float, int> s2 = SigC::retype_return<float>(slot(&f));
 * @endcode
 *
 * SigC::hide_return() is an easy-to-use variant that converts the Slot by
 * dropping its return value, thus converting it to a void slot.
 *
 * Simple Sample usage:
 *
 * @code
 * int f(int);
 * SigC::Slot1<void, int> s = SigC::hide_return( slot(&f) );
 * @endcode
 */

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif



template <class R1,class R2>
struct AdaptorRetypeSlot0_ 
  {
    typedef typename Slot0<R2>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (slot));
      }
  };

/// @ingroup retype
template <class R1,class R2>
Slot0<R1>
  retype(const Slot0<R2> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot0_<R1,R2>::proxy),s);
  }

template <class R1,class R2>
Slot0<R1>
  retype0(const Slot0<R2> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot0_<R1,R2>::proxy),s);
  }



template <class R1,class P1,class R2,class C1>
struct AdaptorRetypeSlot1_ 
  {
    typedef typename Slot1<R2,C1>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class R2,class C1>
Slot1<R1,P1>
  retype(const Slot1<R2,C1> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot1_<R1,P1,R2,C1>::proxy),s);
  }

template <class R1,class P1,class R2,class C1>
Slot1<R1,P1>
  retype1(const Slot1<R2,C1> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot1_<R1,P1,R2,C1>::proxy),s);
  }



template <class R1,class P1,class P2,class R2,class C1,class C2>
struct AdaptorRetypeSlot2_ 
  {
    typedef typename Slot2<R2,C1,C2>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class P2,class R2,class C1,class C2>
Slot2<R1,P1,P2>
  retype(const Slot2<R2,C1,C2> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot2_<R1,P1,P2,R2,C1,C2>::proxy),s);
  }

template <class R1,class P1,class P2,class R2,class C1,class C2>
Slot2<R1,P1,P2>
  retype2(const Slot2<R2,C1,C2> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot2_<R1,P1,P2,R2,C1,C2>::proxy),s);
  }



template <class R1,class P1,class P2,class P3,class R2,class C1,class C2,class C3>
struct AdaptorRetypeSlot3_ 
  {
    typedef typename Slot3<R2,C1,C2,C3>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class P2,class P3,class R2,class C1,class C2,class C3>
Slot3<R1,P1,P2,P3>
  retype(const Slot3<R2,C1,C2,C3> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot3_<R1,P1,P2,P3,R2,C1,C2,C3>::proxy),s);
  }

template <class R1,class P1,class P2,class P3,class R2,class C1,class C2,class C3>
Slot3<R1,P1,P2,P3>
  retype3(const Slot3<R2,C1,C2,C3> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot3_<R1,P1,P2,P3,R2,C1,C2,C3>::proxy),s);
  }



template <class R1,class P1,class P2,class P3,class P4,class R2,class C1,class C2,class C3,class C4>
struct AdaptorRetypeSlot4_ 
  {
    typedef typename Slot4<R2,C1,C2,C3,C4>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class P2,class P3,class P4,class R2,class C1,class C2,class C3,class C4>
Slot4<R1,P1,P2,P3,P4>
  retype(const Slot4<R2,C1,C2,C3,C4> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot4_<R1,P1,P2,P3,P4,R2,C1,C2,C3,C4>::proxy),s);
  }

template <class R1,class P1,class P2,class P3,class P4,class R2,class C1,class C2,class C3,class C4>
Slot4<R1,P1,P2,P3,P4>
  retype4(const Slot4<R2,C1,C2,C3,C4> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot4_<R1,P1,P2,P3,P4,R2,C1,C2,C3,C4>::proxy),s);
  }



template <class R1,class P1,class P2,class P3,class P4,class P5,class R2,class C1,class C2,class C3,class C4,class C5>
struct AdaptorRetypeSlot5_ 
  {
    typedef typename Slot5<R2,C1,C2,C3,C4,C5>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class P2,class P3,class P4,class P5,class R2,class C1,class C2,class C3,class C4,class C5>
Slot5<R1,P1,P2,P3,P4,P5>
  retype(const Slot5<R2,C1,C2,C3,C4,C5> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot5_<R1,P1,P2,P3,P4,P5,R2,C1,C2,C3,C4,C5>::proxy),s);
  }

template <class R1,class P1,class P2,class P3,class P4,class P5,class R2,class C1,class C2,class C3,class C4,class C5>
Slot5<R1,P1,P2,P3,P4,P5>
  retype5(const Slot5<R2,C1,C2,C3,C4,C5> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot5_<R1,P1,P2,P3,P4,P5,R2,C1,C2,C3,C4,C5>::proxy),s);
  }



template <class R1,class P1,class P2,class P3,class P4,class P5,class P6,class R2,class C1,class C2,class C3,class C4,class C5,class C6>
struct AdaptorRetypeSlot6_ 
  {
    typedef typename Slot6<R2,C1,C2,C3,C4,C5,C6>::Proxy Proxy;
    typedef typename Trait<R1>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,p6,slot));
      }
  };

/// @ingroup retype
template <class R1,class P1,class P2,class P3,class P4,class P5,class P6,class R2,class C1,class C2,class C3,class C4,class C5,class C6>
Slot6<R1,P1,P2,P3,P4,P5,P6>
  retype(const Slot6<R2,C1,C2,C3,C4,C5,C6> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot6_<R1,P1,P2,P3,P4,P5,P6,R2,C1,C2,C3,C4,C5,C6>::proxy),s);
  }

template <class R1,class P1,class P2,class P3,class P4,class P5,class P6,class R2,class C1,class C2,class C3,class C4,class C5,class C6>
Slot6<R1,P1,P2,P3,P4,P5,P6>
  retype6(const Slot6<R2,C1,C2,C3,C4,C5,C6> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeSlot6_<R1,P1,P2,P3,P4,P5,P6,R2,C1,C2,C3,C4,C5,C6>::proxy),s);
  }




#if !defined(SIGC_CXX_VOID_CAST_RETURN) && defined(SIGC_CXX_PARTIAL_SPEC)
template <class R2>
struct AdaptorRetypeSlot0_ <void,R2>
  {
    typedef typename Slot0<R2>::Proxy Proxy;
    static void proxy(void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(slot);
      }
  };



template <class P1,class R2,class C1>
struct AdaptorRetypeSlot1_ <void,P1,R2,C1>
  {
    typedef typename Slot1<R2,C1>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,slot);
      }
  };



template <class P1,class P2,class R2,class C1,class C2>
struct AdaptorRetypeSlot2_ <void,P1,P2,R2,C1,C2>
  {
    typedef typename Slot2<R2,C1,C2>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,p2,slot);
      }
  };



template <class P1,class P2,class P3,class R2,class C1,class C2,class C3>
struct AdaptorRetypeSlot3_ <void,P1,P2,P3,R2,C1,C2,C3>
  {
    typedef typename Slot3<R2,C1,C2,C3>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,p2,p3,slot);
      }
  };



template <class P1,class P2,class P3,class P4,class R2,class C1,class C2,class C3,class C4>
struct AdaptorRetypeSlot4_ <void,P1,P2,P3,P4,R2,C1,C2,C3,C4>
  {
    typedef typename Slot4<R2,C1,C2,C3,C4>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,p2,p3,p4,slot);
      }
  };



template <class P1,class P2,class P3,class P4,class P5,class R2,class C1,class C2,class C3,class C4,class C5>
struct AdaptorRetypeSlot5_ <void,P1,P2,P3,P4,P5,R2,C1,C2,C3,C4,C5>
  {
    typedef typename Slot5<R2,C1,C2,C3,C4,C5>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,p2,p3,p4,p5,slot);
      }
  };



template <class P1,class P2,class P3,class P4,class P5,class P6,class R2,class C1,class C2,class C3,class C4,class C5,class C6>
struct AdaptorRetypeSlot6_ <void,P1,P2,P3,P4,P5,P6,R2,C1,C2,C3,C4,C5,C6>
  {
    typedef typename Slot6<R2,C1,C2,C3,C4,C5,C6>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void *data)
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))(p1,p2,p3,p4,p5,p6,slot);
      }
  };



#endif

#ifdef SIGC_CXX_NAMESPACES
} // namespace
#endif

#endif // SIGC_RETYPE_H
