// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_AUTODISCONNECT_H
#define   SIGC_AUTODISCONNECT_H
#include <sigc++/adaptor.h>

/*
  SigC::auto_disconnect
  -------------
  auto_disconnect() causes a slot to disconnect automatically after a
  fixed number of calls.

  Simple Sample usage:

    void f(int);

    Slot1<void,int>   s1=auto_disconnect(slot(&f),2); 
    s1();
    s1(); // slot disconnects
    s1(); 

*/

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// (internal) 
struct AdaptorAutoDisconnectSlotNode : public AdaptorSlotNode
  {
    // struct to post decrement the disconnect count
    struct Dec 
      {
        AdaptorAutoDisconnectSlotNode* node_;
        Dec(AdaptorAutoDisconnectSlotNode& node): node_(&node) {}
        ~Dec() { if (!--node_->count_) node_->notify(false); }
      };
    int count_;
    AdaptorAutoDisconnectSlotNode(FuncPtr proxy,const Node& s,int count);
    virtual ~AdaptorAutoDisconnectSlotNode();
  };




/****************************************************************
***** Adaptor Auto Disconnect Slot, 0 arguments
****************************************************************/
template <class R>
struct AdaptorAutoDisconnectSlot0_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot0<R>::Proxy)(slot->proxy_))
          (slot);
      }
  };

template <class R>
Slot0<R>
  auto_disconnect(const Slot0<R> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot0_<R>::proxy),s,count);
  }


/****************************************************************
***** Adaptor Auto Disconnect Slot, 1 arguments
****************************************************************/
template <class R,class P1>
struct AdaptorAutoDisconnectSlot1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot1<R,P1>::Proxy)(slot->proxy_))
          (p1,slot);
      }
  };

template <class R,class P1>
Slot1<R,P1>
  auto_disconnect(const Slot1<R,P1> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot1_<R,P1>::proxy),s,count);
  }


/****************************************************************
***** Adaptor Auto Disconnect Slot, 2 arguments
****************************************************************/
template <class R,class P1,class P2>
struct AdaptorAutoDisconnectSlot2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot2<R,P1,P2>::Proxy)(slot->proxy_))
          (p1,p2,slot);
      }
  };

template <class R,class P1,class P2>
Slot2<R,P1,P2>
  auto_disconnect(const Slot2<R,P1,P2> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot2_<R,P1,P2>::proxy),s,count);
  }


/****************************************************************
***** Adaptor Auto Disconnect Slot, 3 arguments
****************************************************************/
template <class R,class P1,class P2,class P3>
struct AdaptorAutoDisconnectSlot3_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot3<R,P1,P2,P3>::Proxy)(slot->proxy_))
          (p1,p2,p3,slot);
      }
  };

template <class R,class P1,class P2,class P3>
Slot3<R,P1,P2,P3>
  auto_disconnect(const Slot3<R,P1,P2,P3> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot3_<R,P1,P2,P3>::proxy),s,count);
  }


/****************************************************************
***** Adaptor Auto Disconnect Slot, 4 arguments
****************************************************************/
template <class R,class P1,class P2,class P3,class P4>
struct AdaptorAutoDisconnectSlot4_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot4<R,P1,P2,P3,P4>::Proxy)(slot->proxy_))
          (p1,p2,p3,p4,slot);
      }
  };

template <class R,class P1,class P2,class P3,class P4>
Slot4<R,P1,P2,P3,P4>
  auto_disconnect(const Slot4<R,P1,P2,P3,P4> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot4_<R,P1,P2,P3,P4>::proxy),s,count);
  }


/****************************************************************
***** Adaptor Auto Disconnect Slot, 5 arguments
****************************************************************/
template <class R,class P1,class P2,class P3,class P4,class P5>
struct AdaptorAutoDisconnectSlot5_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<P5>::ref p5,void *data) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((Slot5<R,P1,P2,P3,P4,P5>::Proxy)(slot->proxy_))
          (p1,p2,p3,p4,p5,slot);
      }
  };

template <class R,class P1,class P2,class P3,class P4,class P5>
Slot5<R,P1,P2,P3,P4,P5>
  auto_disconnect(const Slot5<R,P1,P2,P3,P4,P5> &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&AdaptorAutoDisconnectSlot5_<R,P1,P2,P3,P4,P5>::proxy),s,count);
  }



#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_AUTO_DISCONNECT_H
