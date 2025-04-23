// -*- c++ -*-
//   Copyright 2000, Martin Schulze <MHL.Schulze@t-online.de>
//   Copyright 2001, Karl Einar Nelson
/* This is a generated file, do not edit.  Generated from template.macros.m4 */

#ifndef SIGC_HIDE_H
#define SIGC_HIDE_H

/** @defgroup hide
 * SigC::hide() alters a Slot in that it adds one or two parameters
 * whose values are ignored on invocation of the Slot.
 * Thus you can discard one or more of the arguments of a Signal.
 * You have to specify the type of the parameters to ignore as
 * template arguments as in
 * @code
 * SigC::Slot1<void, int> slot1;
 * SigC::Slot0<void> slot2 = SigC::hide<int>(slot1);
 * @endcode
 *
 * SigC::hide_return() alters the Slot by
 * dropping its return value, thus converting it to a void Slot.
 *
 * Simple sample usage:
 *
 * @code
 * int f(int);
 * SigC::Slot1<void, int> s = SigC::hide_return( slot(&f) );
 * @endcode
 */

#include <sigc++/adaptor.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif


template <class R,class H1>
struct AdaptorHide0_1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<H1>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot0<R>::Proxy)(slot->proxy_))(slot);
      }
  };

/// @ingroup hide
template <class H1,class R>
Slot1<R,H1>
hide(const Slot0<R>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide0_1_<R,H1>::proxy), s );
  }


template <class R,class H1,class H2>
struct AdaptorHide0_2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<H1>::ref,typename Trait<H2>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot0<R>::Proxy)(slot->proxy_))(slot);
      }
  };

/// @ingroup hide
template <class H1,class H2,class R>
Slot2<R,H1,H2>
hide(const Slot0<R>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide0_2_<R,H1,H2>::proxy), s );
  }


template <class R,class P1,class H1>
struct AdaptorHide1_1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<H1>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot1<R,P1>::Proxy)(slot->proxy_))(p1,slot);
      }
  };

/// @ingroup hide
template <class H1,class R,class P1>
Slot2<R,P1,H1>
hide(const Slot1<R,P1>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide1_1_<R,P1,H1>::proxy), s );
  }


template <class R,class P1,class H1,class H2>
struct AdaptorHide1_2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<H1>::ref,typename Trait<H2>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot1<R,P1>::Proxy)(slot->proxy_))(p1,slot);
      }
  };

/// @ingroup hide
template <class H1,class H2,class R,class P1>
Slot3<R,P1,H1,H2>
hide(const Slot1<R,P1>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide1_2_<R,P1,H1,H2>::proxy), s );
  }


template <class R,class P1,class P2,class H1>
struct AdaptorHide2_1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<H1>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot2<R,P1,P2>::Proxy)(slot->proxy_))(p1,p2,slot);
      }
  };

/// @ingroup hide
template <class H1,class R,class P1,class P2>
Slot3<R,P1,P2,H1>
hide(const Slot2<R,P1,P2>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide2_1_<R,P1,P2,H1>::proxy), s );
  }


template <class R,class P1,class P2,class H1,class H2>
struct AdaptorHide2_2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<H1>::ref,typename Trait<H2>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot2<R,P1,P2>::Proxy)(slot->proxy_))(p1,p2,slot);
      }
  };

/// @ingroup hide
template <class H1,class H2,class R,class P1,class P2>
Slot4<R,P1,P2,H1,H2>
hide(const Slot2<R,P1,P2>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide2_2_<R,P1,P2,H1,H2>::proxy), s );
  }


template <class R,class P1,class P2,class P3,class H1>
struct AdaptorHide3_1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<H1>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot3<R,P1,P2,P3>::Proxy)(slot->proxy_))(p1,p2,p3,slot);
      }
  };

/// @ingroup hide
template <class H1,class R,class P1,class P2,class P3>
Slot4<R,P1,P2,P3,H1>
hide(const Slot3<R,P1,P2,P3>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide3_1_<R,P1,P2,P3,H1>::proxy), s );
  }


template <class R,class P1,class P2,class P3,class H1,class H2>
struct AdaptorHide3_2_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<H1>::ref,typename Trait<H2>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot3<R,P1,P2,P3>::Proxy)(slot->proxy_))(p1,p2,p3,slot);
      }
  };

/// @ingroup hide
template <class H1,class H2,class R,class P1,class P2,class P3>
Slot5<R,P1,P2,P3,H1,H2>
hide(const Slot3<R,P1,P2,P3>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide3_2_<R,P1,P2,P3,H1,H2>::proxy), s );
  }


template <class R,class P1,class P2,class P3,class P4,class H1>
struct AdaptorHide4_1_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(typename Trait<P1>::ref p1,typename Trait<P2>::ref p2,typename Trait<P3>::ref p3,typename Trait<P4>::ref p4,typename Trait<H1>::ref,void *data) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename Slot4<R,P1,P2,P3,P4>::Proxy)(slot->proxy_))(p1,p2,p3,p4,slot);
      }
  };

/// @ingroup hide
template <class H1,class R,class P1,class P2,class P3,class P4>
Slot5<R,P1,P2,P3,P4,H1>
hide(const Slot4<R,P1,P2,P3,P4>& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&AdaptorHide4_1_<R,P1,P2,P3,P4,H1>::proxy), s );
  }



#ifdef SIGC_CXX_NAMESPACES
}  // namespace SigC
#endif

#endif  // SIGC_HIDE_H
