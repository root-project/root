// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_BIND_RETURN_H
#define   SIGC_BIND_RETURN_H
#include <sigc++/bind.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif



/****************************************************************
***** Adaptor Return Bind Slot 0 arguments
****************************************************************/
template <class R1,class R2>
struct AdaptorBindReturnSlot0_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot0<R2>::Proxy Proxy;
    static R1 proxy(void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2>
Slot0<R1>
  bind_return(const Slot0<R2> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot0_<R1,R2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }


/****************************************************************
***** Adaptor Return Bind Slot 1 arguments
****************************************************************/
template <class R1,class R2,class P1>
struct AdaptorBindReturnSlot1_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot1<R2,P1>::Proxy Proxy;
    static R1 proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2,class P1>
Slot1<R1,P1>
  bind_return(const Slot1<R2,P1> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot1_<R1,R2,P1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }


/****************************************************************
***** Adaptor Return Bind Slot 2 arguments
****************************************************************/
template <class R1,class R2,class P1,class P2>
struct AdaptorBindReturnSlot2_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot2<R2,P1,P2>::Proxy Proxy;
    static R1 proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2,class P1,class P2>
Slot2<R1,P1,P2>
  bind_return(const Slot2<R2,P1,P2> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot2_<R1,R2,P1,P2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }


/****************************************************************
***** Adaptor Return Bind Slot 3 arguments
****************************************************************/
template <class R1,class R2,class P1,class P2,class P3>
struct AdaptorBindReturnSlot3_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot3<R2,P1,P2,P3>::Proxy Proxy;
    static R1 proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2,class P1,class P2,class P3>
Slot3<R1,P1,P2,P3>
  bind_return(const Slot3<R2,P1,P2,P3> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot3_<R1,R2,P1,P2,P3> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }


/****************************************************************
***** Adaptor Return Bind Slot 4 arguments
****************************************************************/
template <class R1,class R2,class P1,class P2,class P3,class P4>
struct AdaptorBindReturnSlot4_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot4<R2,P1,P2,P3,P4>::Proxy Proxy;
    static R1 proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2,class P1,class P2,class P3,class P4>
Slot4<R1,P1,P2,P3,P4>
  bind_return(const Slot4<R2,P1,P2,P3,P4> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot4_<R1,R2,P1,P2,P3,P4> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }


/****************************************************************
***** Adaptor Return Bind Slot 5 arguments
****************************************************************/
template <class R1,class R2,class P1,class P2,class P3,class P4,class P5>
struct AdaptorBindReturnSlot5_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename Slot5<R2,P1,P2,P3,P4,P5>::Proxy Proxy;
    static R1 proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,slot);
        return node.c1_; 
      }
  };

/// @ingroup bind
template <class R1,class R2,class P1,class P2,class P3,class P4,class P5>
Slot5<R1,P1,P2,P3,P4,P5>
  bind_return(const Slot5<R2,P1,P2,P3,P4,P5> &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef AdaptorBindReturnSlot5_<R1,R2,P1,P2,P3,P4,P5> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_BIND_RETURN_H
