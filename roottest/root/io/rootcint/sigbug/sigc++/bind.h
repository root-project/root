// -*- c++ -*-
//   Copyright 2000, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef   SIGC_BIND_H
#define   SIGC_BIND_H
#include <sigc++/adaptor.h>

/** @defgroup bind
 *
 * SigC::bind() alters a SigC::Slot by fixing arguments to certain values.
 *
 * Argument fixing starts from the last argument.
 * Up to two arguments can be bound at a time.
 *
 * Simple sample usage:
 * @code
 * void f(int, int);
 * SigC:Slot2<void, int, int> s1 = SigC::slot(f);
  *
 * SigC::Slot1<void, int>  s2 = SigC::bind(s1,1);
 * s2(2);  // call f with arguments 2,1
 * @endcode
 *
 *  Multibinding usage:
 *
 * @code
 *  void f(int,int);
 *  SigC::Slot2<void, int, int> s1 = SigC::slot(f);
  *
 *  SigC::Slot0<void>  s2 = SigC::bind(s1, 1, 2);
 *  s2();  // call f with arguments 1, 2
 * @endcode
 *
 *  Type specified usage:
 *
 *  @code
 *  class A {};
 *  class B : public A {};
 *  B* b;
 *  SigC::Slot0<void, A*> s1;
 *
 *  SigC::Slot0<void> s2 = SIgC::bind(s1, b);  // B* converted to A*
 * @endcode
 *
 *
 * SigC::bind_return() alters a Slot by fixing the return value to certain values
 *
 * Return value fixing ignores any slot return value.  The slot is
 * destroyed in the process and a new one is created, so references
 * to the slot will no longer be valid.
 *
 * Typecasting may be necessary to match arguments between the
 * slot and the bound return value.  Types must be an exact match.
 * To ensure the proper type, the type can be explicitly specified
 * on template instantation.
 *
 * Simple sample usage:
 * @code
 * void f(int, int);
 * SigC::Slot1<int, int, int>  s1 = SigC::bind_return(slot(&f), 1);
 * std::cout << "s2: " << s1(2, 1) << std::endl;
 * @endcode
 *
 * Type specified usage:
 * @code
 * class A {};
 * class B : public A {};
 * B* b;
 * SigC::Slot1<void> s1;
 *
 * SigC::Slot0<A*> s2 = SigC::bind_return<A*>(s1, b);  // B* must be told to match A*
 * @endcode
 *
 */

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif


/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API AdaptorBindSlotNode : public AdaptorSlotNode
  {
    FuncPtr dtor_;

    AdaptorBindSlotNode(FuncPtr proxy, const Node& s, FuncPtr dtor);

    virtual ~AdaptorBindSlotNode();
  };







template <class C1>
struct AdaptorBindData1_
  {
    typedef AdaptorBindData1_ Self;
    AdaptorBindSlotNode adaptor;
    C1 c1_;
    AdaptorBindData1_(FuncPtr p, const Node& s ,FuncPtr d,
      C1 c1)
    : adaptor(p, s, d), c1_(c1)
      {}

    static void dtor(void* data)
      {
        Self& node = *reinterpret_cast<Self*>(data);
        node.c1_.~C1();
      }
  }; 


template <class C1,class C2>
struct AdaptorBindData2_
  {
    typedef AdaptorBindData2_ Self;
    AdaptorBindSlotNode adaptor;
    C1 c1_;
        C2 c2_;
    AdaptorBindData2_(FuncPtr p, const Node& s ,FuncPtr d,
      C1 c1,C2 c2)
    : adaptor(p, s, d), c1_(c1),c2_(c2)
      {}

    static void dtor(void* data)
      {
        Self& node = *reinterpret_cast<Self*>(data);
        node.c1_.~C1();
        node.c2_.~C2();
      }
  }; 



template <class R,class C1>
struct AdaptorBindSlot0_1_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot1<R,C1>::Proxy Proxy;
    static RType proxy(void *data) 
      { 
        typedef AdaptorBindData1_<C1> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (node.c1_,slot);
      }

  };

/// @ingroup bind
template <class A1,class R,class C1>
Slot0<R>
  bind(const Slot1<R,C1>& s,
       A1 a1)
  { 
    typedef AdaptorBindData1_<C1> Data;
    typedef AdaptorBindSlot0_1_<R,C1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1));
  }


template <class R,class P1,class C1>
struct AdaptorBindSlot1_1_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot2<R,P1,C1>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        typedef AdaptorBindData1_<C1> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,node.c1_,slot);
      }

  };

/// @ingroup bind
template <class A1,class R,class P1,class C1>
Slot1<R,P1>
  bind(const Slot2<R,P1,C1>& s,
       A1 a1)
  { 
    typedef AdaptorBindData1_<C1> Data;
    typedef AdaptorBindSlot1_1_<R,P1,C1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1));
  }


template <class R,class P1,class P2,class C1>
struct AdaptorBindSlot2_1_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot3<R,P1,P2,C1>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        typedef AdaptorBindData1_<C1> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,p2,node.c1_,slot);
      }

  };

/// @ingroup bind
template <class A1,class R,class P1,class P2,class C1>
Slot2<R,P1,P2>
  bind(const Slot3<R,P1,P2,C1>& s,
       A1 a1)
  { 
    typedef AdaptorBindData1_<C1> Data;
    typedef AdaptorBindSlot2_1_<R,P1,P2,C1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1));
  }


template <class R,class P1,class P2,class P3,class C1>
struct AdaptorBindSlot3_1_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot4<R,P1,P2,P3,C1>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        typedef AdaptorBindData1_<C1> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,p2,p3,node.c1_,slot);
      }

  };

/// @ingroup bind
template <class A1,class R,class P1,class P2,class P3,class C1>
Slot3<R,P1,P2,P3>
  bind(const Slot4<R,P1,P2,P3,C1>& s,
       A1 a1)
  { 
    typedef AdaptorBindData1_<C1> Data;
    typedef AdaptorBindSlot3_1_<R,P1,P2,P3,C1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1));
  }


template <class R,class P1,class P2,class P3,class P4,class C1>
struct AdaptorBindSlot4_1_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot5<R,P1,P2,P3,P4,C1>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,void *data) 
      { 
        typedef AdaptorBindData1_<C1> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,p2,p3,p4,node.c1_,slot);
      }

  };

/// @ingroup bind
template <class A1,class R,class P1,class P2,class P3,class P4,class C1>
Slot4<R,P1,P2,P3,P4>
  bind(const Slot5<R,P1,P2,P3,P4,C1>& s,
       A1 a1)
  { 
    typedef AdaptorBindData1_<C1> Data;
    typedef AdaptorBindSlot4_1_<R,P1,P2,P3,P4,C1> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1));
  }



template <class R,class C1,class C2>
struct AdaptorBindSlot0_2_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot2<R,C1,C2>::Proxy Proxy;
    static RType proxy(void *data) 
      { 
        typedef AdaptorBindData2_<C1,C2> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (node.c1_,node.c2_,slot);
      }

  };

/// @ingroup bind
template <class A1,class A2,class R,class C1,class C2>
Slot0<R>
  bind(const Slot2<R,C1,C2>& s,
       A1 a1,A2 a2)
  { 
    typedef AdaptorBindData2_<C1,C2> Data;
    typedef AdaptorBindSlot0_2_<R,C1,C2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1,a2));
  }


template <class R,class P1,class C1,class C2>
struct AdaptorBindSlot1_2_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot3<R,P1,C1,C2>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,void *data) 
      { 
        typedef AdaptorBindData2_<C1,C2> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,node.c1_,node.c2_,slot);
      }

  };

/// @ingroup bind
template <class A1,class A2,class R,class P1,class C1,class C2>
Slot1<R,P1>
  bind(const Slot3<R,P1,C1,C2>& s,
       A1 a1,A2 a2)
  { 
    typedef AdaptorBindData2_<C1,C2> Data;
    typedef AdaptorBindSlot1_2_<R,P1,C1,C2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1,a2));
  }


template <class R,class P1,class P2,class C1,class C2>
struct AdaptorBindSlot2_2_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot4<R,P1,P2,C1,C2>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,void *data) 
      { 
        typedef AdaptorBindData2_<C1,C2> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,p2,node.c1_,node.c2_,slot);
      }

  };

/// @ingroup bind
template <class A1,class A2,class R,class P1,class P2,class C1,class C2>
Slot2<R,P1,P2>
  bind(const Slot4<R,P1,P2,C1,C2>& s,
       A1 a1,A2 a2)
  { 
    typedef AdaptorBindData2_<C1,C2> Data;
    typedef AdaptorBindSlot2_2_<R,P1,P2,C1,C2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1,a2));
  }


template <class R,class P1,class P2,class P3,class C1,class C2>
struct AdaptorBindSlot3_2_
  {
    typedef typename Trait<R>::type RType;
    typedef typename Slot5<R,P1,P2,P3,C1,C2>::Proxy Proxy;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,void *data) 
      { 
        typedef AdaptorBindData2_<C1,C2> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (p1,p2,p3,node.c1_,node.c2_,slot);
      }

  };

/// @ingroup bind
template <class A1,class A2,class R,class P1,class P2,class P3,class C1,class C2>
Slot3<R,P1,P2,P3>
  bind(const Slot5<R,P1,P2,P3,C1,C2>& s,
       A1 a1,A2 a2)
  { 
    typedef AdaptorBindData2_<C1,C2> Data;
    typedef AdaptorBindSlot3_2_<R,P1,P2,P3,C1,C2> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),a1,a2));
  }



#ifdef SIGC_CXX_NAMESPACES
} // namespace
#endif

#endif // SIGC_BIND_H
