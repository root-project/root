// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_RETYPE_RETURN_H
#define   SIGC_RETYPE_RETURN_H
#include <sigc++/adaptor.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif



/****************************************************************
***** Adaptor Return Type Slot, 0 arguments
****************************************************************/
template <class R2,class R1>
struct AdaptorRetypeReturnSlot0_ 
  {
    typedef typename Slot0<R1>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (slot));
      }
  };

/// @ingroup retype
template <class R2,class R>
Slot0<R2>
  retype_return(const Slot0<R> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot0_<R2,R>::proxy),s);
  }

/// @ingroup hide
template <class R>
Slot0<void>
  hide_return(const Slot0<R> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 1 arguments
****************************************************************/
template <class R2,class R1,class P1>
struct AdaptorRetypeReturnSlot1_ 
  {
    typedef typename Slot1<R1,P1>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1>
Slot1<R2,P1>
  retype_return(const Slot1<R,P1> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot1_<R2,R,P1>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1>
Slot1<void,P1>
  hide_return(const Slot1<R,P1> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 2 arguments
****************************************************************/
template <class R2,class R1,class P1,class P2>
struct AdaptorRetypeReturnSlot2_ 
  {
    typedef typename Slot2<R1,P1,P2>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1,class P2>
Slot2<R2,P1,P2>
  retype_return(const Slot2<R,P1,P2> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot2_<R2,R,P1,P2>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1,class P2>
Slot2<void,P1,P2>
  hide_return(const Slot2<R,P1,P2> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 3 arguments
****************************************************************/
template <class R2,class R1,class P1,class P2,class P3>
struct AdaptorRetypeReturnSlot3_ 
  {
    typedef typename Slot3<R1,P1,P2,P3>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1,class P2,class P3>
Slot3<R2,P1,P2,P3>
  retype_return(const Slot3<R,P1,P2,P3> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot3_<R2,R,P1,P2,P3>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1,class P2,class P3>
Slot3<void,P1,P2,P3>
  hide_return(const Slot3<R,P1,P2,P3> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 4 arguments
****************************************************************/
template <class R2,class R1,class P1,class P2,class P3,class P4>
struct AdaptorRetypeReturnSlot4_ 
  {
    typedef typename Slot4<R1,P1,P2,P3,P4>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1,class P2,class P3,class P4>
Slot4<R2,P1,P2,P3,P4>
  retype_return(const Slot4<R,P1,P2,P3,P4> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot4_<R2,R,P1,P2,P3,P4>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1,class P2,class P3,class P4>
Slot4<void,P1,P2,P3,P4>
  hide_return(const Slot4<R,P1,P2,P3,P4> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 5 arguments
****************************************************************/
template <class R2,class R1,class P1,class P2,class P3,class P4,class P5>
struct AdaptorRetypeReturnSlot5_ 
  {
    typedef typename Slot5<R1,P1,P2,P3,P4,P5>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1,class P2,class P3,class P4,class P5>
Slot5<R2,P1,P2,P3,P4,P5>
  retype_return(const Slot5<R,P1,P2,P3,P4,P5> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot5_<R2,R,P1,P2,P3,P4,P5>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1,class P2,class P3,class P4,class P5>
Slot5<void,P1,P2,P3,P4,P5>
  hide_return(const Slot5<R,P1,P2,P3,P4,P5> &s)
  {
    return retype_return<void>(s);
  }


/****************************************************************
***** Adaptor Return Type Slot, 6 arguments
****************************************************************/
template <class R2,class R1,class P1,class P2,class P3,class P4,class P5,class P6>
struct AdaptorRetypeReturnSlot6_ 
  {
    typedef typename Slot6<R1,P1,P2,P3,P4,P5,P6>::Proxy Proxy;
    typedef typename Trait<R2>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return RType(((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,p6,slot));
      }
  };

/// @ingroup retype
template <class R2,class R,class P1,class P2,class P3,class P4,class P5,class P6>
Slot6<R2,P1,P2,P3,P4,P5,P6>
  retype_return(const Slot6<R,P1,P2,P3,P4,P5,P6> &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&AdaptorRetypeReturnSlot6_<R2,R,P1,P2,P3,P4,P5,P6>::proxy),s);
  }

/// @ingroup hide
template <class R,class P1,class P2,class P3,class P4,class P5,class P6>
Slot6<void,P1,P2,P3,P4,P5,P6>
  hide_return(const Slot6<R,P1,P2,P3,P4,P5,P6> &s)
  {
    return retype_return<void>(s);
  }



#if !defined(SIGC_CXX_VOID_CAST_RETURN) && defined(SIGC_CXX_PARTIAL_SPEC)
/****************************************************************
***** Adaptor Return Type Slot, 0 arguments
****************************************************************/
template <class R1>
struct AdaptorRetypeReturnSlot0_ <void,R1>
  {
    typedef typename Slot0<R1>::Proxy Proxy;
    static void proxy(void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 1 arguments
****************************************************************/
template <class R1,class P1>
struct AdaptorRetypeReturnSlot1_ <void,R1,P1>
  {
    typedef typename Slot1<R1,P1>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 2 arguments
****************************************************************/
template <class R1,class P1,class P2>
struct AdaptorRetypeReturnSlot2_ <void,R1,P1,P2>
  {
    typedef typename Slot2<R1,P1,P2>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 3 arguments
****************************************************************/
template <class R1,class P1,class P2,class P3>
struct AdaptorRetypeReturnSlot3_ <void,R1,P1,P2,P3>
  {
    typedef typename Slot3<R1,P1,P2,P3>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 4 arguments
****************************************************************/
template <class R1,class P1,class P2,class P3,class P4>
struct AdaptorRetypeReturnSlot4_ <void,R1,P1,P2,P3,P4>
  {
    typedef typename Slot4<R1,P1,P2,P3,P4>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 5 arguments
****************************************************************/
template <class R1,class P1,class P2,class P3,class P4,class P5>
struct AdaptorRetypeReturnSlot5_ <void,R1,P1,P2,P3,P4,P5>
  {
    typedef typename Slot5<R1,P1,P2,P3,P4,P5>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,slot);
      }
  };



/****************************************************************
***** Adaptor Return Type Slot, 6 arguments
****************************************************************/
template <class R1,class P1,class P2,class P3,class P4,class P5,class P6>
struct AdaptorRetypeReturnSlot6_ <void,R1,P1,P2,P3,P4,P5,P6>
  {
    typedef typename Slot6<R1,P1,P2,P3,P4,P5,P6>::Proxy Proxy;
    static void proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,typename Trait<P6>::ref p6,void *data) 
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,p6,slot);
      }
  };



#endif

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_RETYPE_RETURN_H
